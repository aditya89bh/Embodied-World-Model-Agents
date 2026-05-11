ACTIONS = ['right', 'down', 'left', 'up']
GOAL = (4, 4)


class RolloutPlanner:
    def __init__(self, model, belief_map=None, actions=None, horizon=3):
        self.model = model
        self.belief_map = belief_map
        self.actions = actions or ACTIONS
        self.horizon = horizon

    def choose_action(self, start_state):
        candidates = []

        for action in self.actions:
            rollout = self._rollout(start_state, action)
            score_breakdown = self._score_rollout(rollout)
            candidates.append({
                'action': action,
                'score': score_breakdown['total'],
                'score_breakdown': score_breakdown,
                'rollout': rollout,
            })

        candidates.sort(key=lambda item: item['score'], reverse=True)
        return candidates[0], candidates

    def _rollout(self, start_state, first_action):
        state = start_state
        trace = []

        for step in range(self.horizon):
            action = first_action if step == 0 else self._greedy_goal_action(state)
            next_state = self.model.predict(state, action)
            confidence = self.model.confidence(state, action)
            uncertainty = self.model.uncertainty(state, action)
            distribution = self.model.distribution(state, action)
            information_gain = self._information_gain(next_state)

            trace.append({
                'state': state,
                'action': action,
                'next_state': next_state,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'distribution': distribution,
                'information_gain': information_gain,
            })

            state = next_state

        return trace

    def _score_rollout(self, rollout):
        final_state = rollout[-1]['next_state']
        distance_score = -self._manhattan(final_state, GOAL)
        confidence_bonus = sum(step['confidence'] for step in rollout) / len(rollout)
        uncertainty_penalty = sum(step['uncertainty'] for step in rollout) / len(rollout)
        exploration_bonus = sum(step['information_gain'] for step in rollout) * 0.25
        movement_bonus = sum(
            1 for step in rollout if step['state'] != step['next_state']
        ) * 0.1

        total = (
            distance_score
            + confidence_bonus
            + movement_bonus
            + exploration_bonus
            - uncertainty_penalty
        )

        return {
            'distance_score': distance_score,
            'confidence_bonus': confidence_bonus,
            'movement_bonus': movement_bonus,
            'exploration_bonus': exploration_bonus,
            'uncertainty_penalty': uncertainty_penalty,
            'total': total,
        }

    def _information_gain(self, state):
        if self.belief_map is None:
            return 0
        return self.belief_map.information_gain(state)

    def _greedy_goal_action(self, state):
        x, y = state
        gx, gy = GOAL

        if x < gx:
            return 'right'
        if y < gy:
            return 'down'
        if x > gx:
            return 'left'
        if y > gy:
            return 'up'
        return 'stay'

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
