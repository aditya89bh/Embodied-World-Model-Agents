def render_grid(agent_position, hidden_block, width=5, height=5):
    rows = []

    for y in range(height):
        row = []
        for x in range(width):
            pos = (x, y)

            if pos == agent_position:
                row.append('A')
            elif pos == hidden_block:
                row.append('X')
            elif pos == (4, 4):
                row.append('G')
            else:
                row.append('.')

        rows.append(' '.join(row))

    return '\n'.join(rows)
