def shift_coordinates(coords):
    min_x = min(x for x, y in coords)
    min_y = min(y for x, y in coords)

    # Shift all points to make min_x and min_y at least 0
    new_coords = [(x - min_x, y - min_y) for x, y in coords]

    return new_coords

# Example Usage
input_coords = [(1, -2), (-2, 4), (-1, -1), (-8, -3), (0, 4), (10, -3)]
output_coords = shift_coordinates(input_coords)
print(output_coords)
