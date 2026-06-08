import numpy as np
import math
import random
from scipy.linalg import solve_triangular
from planners.base import get_path_cost, BasePlanner, get_sampling_cost
from planners.deployment_time import get_iig_deployment_time


class IIGNode:
    def __init__(self, state, cost_from_root=0.0, info_from_root=0.0, parent=None,
                 path_indices=None, gp_L=None, gp_X=None, gp_y=None, gp_alpha=None, gp_w=None):
        self.state = state
        self.cost_from_root = cost_from_root
        # info_from_root stores PURE cumulative information (not cost-normalized reward).
        # This is I_new in the paper — the sum of raw GP variance scores along the path.
        # The paper keeps information and cost as completely separate quantities on each
        # node (Algorithm 2, lines 20-21: I_new and C_new are tracked independently).
        self.info_from_root = info_from_root
        self.parent = parent
        self.path_indices = path_indices if path_indices is not None else set()

        # Dynamic GP state for IIG
        self.gp_L = gp_L
        self.gp_X = gp_X
        self.gp_y = gp_y
        self.gp_alpha = gp_alpha
        self.gp_w = gp_w


class IIGPlanner(BasePlanner):
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42,
                 normalize_reward=True, optimized_noise_var=0.03, **kwargs):
        super().__init__(size, df, gp=gp, output_dir=output_dir, inf_criteria=inf_criteria,
                         seed=seed, normalize_reward=normalize_reward,
                         optimized_noise_var=optimized_noise_var, **kwargs)
        if not self.is_dynamic:
            raise ValueError("IIGPlanner requires is_dynamic=True.")
        self.gp = gp
        self.samples = 5000
        self.step_limit = int(kwargs.get('horizon', 3))
        self.ric_threshold = float(kwargs.get('ric_threshold', 0.05))

        # Convergence monitoring
        self.ric_window = []
        self.window_size = 50
        self.is_converged = False
        self.rng = random.Random(self.seed)

    def select_next(self, inf_map, sampled_indices):
        self.deployment_time = get_iig_deployment_time(
            len(sampled_indices), self.inf_criteria, self.step_limit, self.ric_threshold, self.size_cat
        )
        self.step_budget = self.compute_step_budget(len(sampled_indices))
        self.prepare_step()

        curr_row = self.df.iloc[sampled_indices[-1]]
        root_state = (curr_row['x'], curr_row['y'])

        root_idx = int(round(root_state[1])) * self.size + int(round(root_state[0]))
        root = IIGNode(
            root_state,
            cost_from_root=0.0,
            info_from_root=0.0,          # Root has zero accumulated information
            path_indices={root_idx},
            gp_L=self.gp.L_, gp_X=self.gp.X_train_,
            gp_y=self.gp.y_train_, gp_alpha=self.gp.alpha_,
            gp_w=getattr(self, 'w_ref_root', None)
        )

        nodes = [root]
        global_history = set(sampled_indices)
        best_leaf = root
        max_info = -1.0

        # CHANGE 1: Per-node sample counter (replaces global loop index `i`).
        # The paper (Definition 4, Algorithm 2 lines 9/26) defines n_sample as the number
        # of draws it takes to successfully add ONE new node. It resets to zero immediately
        # after a node is added (Algorithm 2 line 26). Using the global loop index `i+1`
        # was wrong because it never resets, making IRIC shrink monotonically toward zero
        # regardless of how informative new nodes are, causing premature convergence.
        nsample_since_last_node = 0

        termination_reason = "max_samples"

        for i in range(self.samples):
            rand_state = (self.rng.uniform(0, self.size - 1), self.rng.uniform(0, self.size - 1))
            nearest_node = min(nodes, key=lambda n: math.dist(n.state, rand_state))

            # CHANGE 2: Each neighbor steers independently toward x_feasible.
            # The paper's Algorithm 2 (line 16) calls Steer(x_near, x_feasible, Δ) inside
            # the neighbor loop, meaning each neighbor can produce a *different* x_new.
            # The original code called steer_by_cost once outside the loop and reused the
            # same new_state for every neighbor — a simplification that changes which nodes
            # are actually added and can miss valid extensions from non-nearest neighbors.
            x_feasible_state, _ = self.steer_by_cost(nearest_node.state, rand_state)

            search_radius = self.step_budget * 1.5
            neighbors = [n for n in nodes if math.dist(n.state, x_feasible_state) <= search_radius]

            # CHANGE 1 (continued): Increment BEFORE the neighbor loop so that every
            # sample attempt — including those that produce no new node — is counted.
            # This matches Algorithm 2 line 9: nsample increments once per outer iteration.
            nsample_since_last_node += 1

            node_added_this_sample = False

            for neighbor in neighbors:
                # CHANGE 2 (continued): Steer each neighbor independently toward x_feasible.
                new_state, _ = self.steer_by_cost(neighbor.state, x_feasible_state)

                tx, ty = int(round(new_state[0])), int(round(new_state[1]))
                target_idx = ty * self.size + tx

                # Skip if this branch or globally has already visited this cell.
                if target_idx in neighbor.path_indices:
                    continue
                if target_idx in global_history:
                    continue

                # Compute raw GP variance score for this new point given parent's belief.
                inf_score = self.get_inf_score(new_state, neighbor.gp_L, neighbor.gp_X,
                                               neighbor.gp_w, inf_map=inf_map)

                edge_cost = get_path_cost(
                    neighbor.state[0], neighbor.state[1],
                    new_state[0], new_state[1],
                    self.slope_grid, self.elevation_grid,
                    self.size, len(sampled_indices), self.deployment_time
                )
                if edge_cost <= 0:
                    continue

                # CHANGE 3: info_from_root accumulates PURE information, not reward.
                # The paper separates I_new (information) and C_new (cost) on every node
                # (Algorithm 2 lines 20-21). Mixing them by dividing inf_score by edge_cost
                # before accumulating corrupted the RIC ratio because the denominator
                # (I_near) would contain cost-scaled values rather than raw information
                # values, making the ratio dimensionally inconsistent with Definition 3.
                #
                # Cost-normalization is still used for SELECTING the best parent (we want
                # the most information-efficient path), but the stored info_from_root value
                # must remain a pure information quantity so that the RIC ratio is meaningful.
                info_gain_new = neighbor.info_from_root + inf_score  # pure information

                # Cost-normalized utility used only for parent selection (not stored).
                if self.normalize_reward:
                    selection_utility = neighbor.info_from_root + (inf_score / edge_cost)
                else:
                    selection_utility = info_gain_new

                # CHANGE 4: Track the best candidate across all neighbors in this sample.
                # We collect the best (parent, new_state, info, cost) tuple and add exactly
                # one node per successful sample iteration, matching Algorithm 2's structure
                # where the inner for-loop feeds candidates but the graph grows one node at
                # a time per outer iteration.
                if not node_added_this_sample or selection_utility > best_selection_utility:
                    best_selection_utility = selection_utility
                    best_new_state = new_state
                    best_target_idx = target_idx
                    best_parent = neighbor
                    best_inf_score = inf_score
                    best_info_gain_new = info_gain_new
                    best_edge_cost = edge_cost
                    node_added_this_sample = True

            if node_added_this_sample:
                # Add exactly one new node for this sample iteration.
                new_path = best_parent.path_indices.copy()
                new_path.add(best_target_idx)

                k_star = self.gp.kernel_(best_parent.gp_X, np.atleast_2d(best_new_state))
                pred_mean = (k_star.T @ best_parent.gp_alpha)[0]

                new_L, new_X, new_y, new_alpha, new_w = self.update_gp_and_w(
                    best_parent.gp_L, best_parent.gp_X, best_parent.gp_y,
                    best_parent.gp_alpha, best_parent.gp_w, best_new_state, pred_mean
                )

                new_node = IIGNode(
                    best_new_state,
                    cost_from_root=best_parent.cost_from_root + best_edge_cost,
                    # CHANGE 3 (continued): Store pure accumulated information.
                    info_from_root=best_info_gain_new,
                    parent=best_parent,
                    path_indices=new_path,
                    gp_L=new_L, gp_X=new_X, gp_y=new_y,
                    gp_alpha=new_alpha, gp_w=new_w
                )
                nodes.append(new_node)

                # CHANGE 5: Correct IRIC calculation (Definitions 3 & 4 from the paper).
                #
                # Paper Definition 3 — RIC:
                #   RIC = (I_new / I_near) - 1
                # where I_new and I_near are the cumulative information values of the new
                # node and its parent respectively.
                #
                # Paper Definition 4 — Penalized IRIC:
                #   IRIC = RIC / n_sample
                # where n_sample is the count of draw attempts since the last successful
                # node addition (resets to zero after each addition).
                #
                # The original code used:
                #   ric = best_node_gain / best_parent.info_from_root   (wrong numerator)
                #   penalized_ric = ric / (i + 1)                       (never-resetting denom)
                #
                # Both were incorrect. The numerator should be the ratio of the NEW NODE's
                # cumulative info to its PARENT's cumulative info, minus 1 — not a ratio of
                # the raw edge score to the parent's info. The denominator must use the
                # resetting per-node counter so that nodes found quickly (high IRIC) are
                # distinguished from nodes found only after many failed draws (low IRIC).
                I_near = best_parent.info_from_root + 1e-9  # avoid division by zero at root
                ric = (best_info_gain_new / I_near) - 1.0
                iric = ric / max(nsample_since_last_node, 1)

                # CHANGE 1 (reset): Reset the per-node sample counter NOW, after the node
                # is successfully added, exactly as Algorithm 2 line 26 specifies.
                nsample_since_last_node = 0

                self.ric_window.append(iric)
                if len(self.ric_window) > self.window_size:
                    self.ric_window.pop(0)

                if len(self.ric_window) == self.window_size:
                    if np.mean(self.ric_window) < self.ric_threshold:
                        self.is_converged = True
                        termination_reason = "ric_threshold"
                        break

                # Track best leaf by pure information (not cost-normalized utility).
                if best_info_gain_new > max_info:
                    max_info = best_info_gain_new
                    best_leaf = new_node

        # Final convergence check after loop exhaustion.
        if len(self.ric_window) == self.window_size and np.mean(self.ric_window) < self.ric_threshold:
            self.is_converged = True
            termination_reason = "ric_threshold"  # <-- add this line

        if best_leaf == root:
            return self.find_closest_unvisited(root_idx, global_history)
        
        print(f"[IIGPlanner] Terminated: {termination_reason}")

        # CHANGE 6: Walk up to the first-step child of root along the best-leaf's path.
        # This is unchanged in mechanics but is now operating on a correctly structured
        # tree, so the traversal reliably returns the immediate next waypoint.
        curr = best_leaf
        while curr.parent and curr.parent.parent:
            curr = curr.parent

        return int(round(curr.state[1])) * self.size + int(round(curr.state[0]))
    def find_closest_unvisited(self, start_idx, global_history):
        """BFS to find the nearest grid cell not in history."""
        queue = [start_idx]
        visited_in_search = {start_idx}
        while queue:
            curr = queue.pop(0)
            if curr not in global_history:
                return curr
            cx, cy = curr % self.size, curr // self.size
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    n_idx = ny * self.size + nx
                    if n_idx not in visited_in_search:
                        visited_in_search.add(n_idx)
                        queue.append(n_idx)
        return start_idx
