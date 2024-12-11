import torch
import numpy as np
import random

piece_names = "IJLOSTZ"
piece_dims = [4, 3, 3, 2, 3, 3, 3]
piece_rots = [2, 4, 4, 1, 2, 4, 2]
piece_shapes = [
    [
        [0,0,0,0],
        [1,1,1,1],
        [0,0,0,0],
        [0,0,0,0]
    ],
    [
        [1,0,0],
        [1,1,1],
        [0,0,0]
    ],
    [
        [0,0,1],
        [1,1,1],
        [0,0,0]
    ],
    [
        [1,1],
        [1,1]
    ],
    [
        [0,1,1],
        [1,1,0],
        [0,0,0]
    ],
    [
        [0,1,0],
        [1,1,1],
        [0,0,0]
    ],
    [
        [1,1,0],
        [0,1,1],
        [0,0,0]
    ]
]
piece_shape_rots = []
piece_box_rots = []

def initialize():
    if len(piece_shape_rots) > 0:
        return
    
    for ind in range(7):
        rots = []
        boxs = []
        dim = piece_dims[ind]
        for rot in range(piece_rots[ind]):
            shape = piece_shapes[ind]
            for _iter in range(rot):
                new_shape = [[0]*dim for i in range(dim)]
                for x in range(dim):
                    for y in range(dim):
                        new_shape[dim-1-y][x] = shape[x][y]
                shape = new_shape
            rots.append(shape)

            # find bounding box
            box = [dim, -1, dim, -1]
            for x in range(dim):
                for y in range(dim):
                    if shape[x][y] == 1:
                        box[0] = min(box[0], x)
                        box[1] = max(box[1], x)
                        box[2] = min(box[2], y)
                        box[3] = max(box[3], y)
            boxs.append(box)
        piece_shape_rots.append(rots)
        piece_box_rots.append(boxs)

def extract_info(state):
    grid = ["" for i in range(20)]
    for row in range(20):
        for col in range(10):
            if state[row*10 + col] == 1:
                grid[row] += "O"
            else:
                grid[row] += "_"

    pieces = ""
    for i in range(7):
        ind = np.argmax(state[200 + i*7 : 200 + (i+1)*7])
        if np.sum(state[200 + i*7 : 200 + (i+1)*7]) > 1:
            print("alert", i, np.sum(state[200 + i*7 : 200 + (i+1)*7]))
        pieces += piece_names[ind]

    return grid, pieces

def print_info(state):
    grid, pieces = extract_info(state)
    for row in grid:
        print(row)
    print("Current:", pieces[0])
    print("Held:", pieces[6])
    print("Queue:", pieces[1:6])
    print("")

def transition(state, action, height = []):
    """
    Life is pleasant.
    Death is peaceful.
    It's the transition that's troublesome.
                             - Isaac Asimov 
    """
    NUM_PIECES = 7
    QUEUE_SIZE = 5
    NUM_ACTIONS = 80 # change this number later

    state = np.copy(state)
    # drop piece first to update board, return if failed
    state = drop_piece(state, action, height)
    if state is None:
        return None, 0, True

    state, cleared, done = clear_lines(state)

    # Swap held piece
    SWAP = (action >= NUM_ACTIONS / 2)
    if SWAP:
        tmp = state[242:249]
        state[242:249] = state[200:207]
        state[200:207] = tmp

    # Update queue
    current_pieces = []
    for i in reversed(range(QUEUE_SIZE)):
        start = -((i + 3) * NUM_PIECES)
        state[start:start+7] = state[start+7:start+14]
        # Grab current piece values
        arg = np.argmax(state[start:start+7])
        current_pieces.append(arg)

    # Add next piece to queue - not counting current piece bc that would make it deterministic
    remainder = [i for i in range(7) if i not in current_pieces]
    next_indicator = random.choice(remainder)
    next_piece = np.zeros(NUM_PIECES, dtype='uint8')
    next_piece[next_indicator] = 1
    state[-14:-7] = next_piece

    return state, cleared, done

def drop_piece(state, action, height = []):
    # extract piece information
    # print(state, action)
    ind = np.argmax(state[200:207])
    if action >= 40:
        ind = np.argmax(state[242:249])

    # extract column and rotation from action number
    col = -1 + (action % 10)
    rot = (action // 10) % 4

    if rot >= piece_rots[ind]:
        return None

    dim = piece_dims[ind]
    shape = piece_shape_rots[ind][rot]
    box = piece_box_rots[ind][rot]
    
    # bad column
    if col + box[2] < 0 or col + box[3] >= 10:
        return None 
        
    cur_row = 19

    if len(height) == 0:
        height = [20]*10
        for y in range(col + box[2], col + box[3]+1):
            while height[y] > 0 and state[(20 - height[y])*10 + y] == 0:
                height[y] -= 1 

    for x in range(dim):
        for y in range(dim):
            if shape[x][y] == 1:
                cur_row = min(cur_row, 19 - height[col + y] - x)

    if cur_row + box[0] < 0:
        return None
    
    for x in range(dim):
        for y in range(dim):
            if shape[x][y] == 1:
                state[(x+cur_row)*10 + (col+y)] = 1

    return state

def clear_lines(state):
    cleared = 0
    done = False
    for row in range(19, -1, -1):
        if np.sum(state[row*10 : (row+1)*10]) == 10:
            cleared += 1
            if row == 19:
                done = True
        elif cleared != 0:
            state[(row+cleared)*10 : (row+cleared+1)*10] = state[row*10 : (row+1)*10]
    state[0 : cleared*10] = np.zeros(cleared*10, dtype = 'uint8')
    return state, cleared, done

def starting_position(row_length=10, num_rows=20, starting_rows=10, next_pieces=5):
    """
    Generate a random Tetris board represented by 250 indicators (0 or 1),
    with the following specifications:
    - The first 200 entries represent the board, with each 10-entry chunk
    representing a row from top to bottom
    - 0 means the cell is empty, 1 means the cell is taken by a block
    - Only the bottom 10 rows are initialized, with 1 hole per row and no
    adjacent holes
    - The next 7 entries are a one-hot encoding of which piece is the
    current piece
    - The next 35 entries are one-hot encodings of which pieces are the
    next 5 queued pieces
    - The last 8 entries are a one-hot encoding of which piece is held
    (can also be empty)
    
    Returns:
        np.Array: A 1D array of size 250 meeting the above constraints.
    """
    # Initialize variables
    num_pieces = 7  # total number of possible tetris pieces
    my_pieces = next_pieces + 2  # add for current and held piece
    total_indicators = row_length * num_rows + my_pieces * 7
    total_cells = row_length * num_rows
    starting_holes = row_length * (num_rows - starting_rows)

    # Create a tensor of size 250 initialized to zeros
    state = np.zeros(total_indicators, dtype='uint8')
    
    # Populate bottom 10 rows
    prev_hole = -1
    for i in range(starting_rows):  
        # Start with all ones
        row = np.ones(row_length, dtype='uint8')  

        # Determine index of hole in current row
        possible_values = np.array([i for i in range(row_length) if i != prev_hole], dtype='uint8')
        next_hole = possible_values[torch.randint(0, len(possible_values), (1,))].item()

        # Create and set row
        row[next_hole] = 0
        state[starting_holes + i * row_length:starting_holes + (i + 1) * row_length] = row 
        prev_hole = next_hole

    # Get random order of starting pieces
    sample = torch.randperm(num_pieces)

    # Populate starting pieces (now including held piece)
    for i in range(my_pieces):
        # Start with all zeros
        row = np.zeros(num_pieces, dtype='uint8')

        # Indicate which piece and add row
        row[sample[i]] = 1
        state[total_cells + i * num_pieces:total_cells + (i + 1) * num_pieces] = row 
    
    return state

def get_heights(state):
    height = [20]*10
    for y in range(10):
        while height[y] > 0 and state[(20 - height[y])*10 + y] == 0:
            height[y] -= 1 
    return height

# if __name__ == "__main__":
#     initialize()
#     state = starting_position()
#     print_info(state)
#     moves = 0
#     while True:
#         best_score, best_act = -1e9, -1
#         for act in range(80):
#             cur_score = -1e9
#             nxt, cleared, done = transition(state, act)
#             if nxt is None:
#                 continue
                
#             cur_score = 0.760777 * cleared + eval_board(nxt)
#             # for act2 in range(80):
#             #     nxt2, cleared2, done2 = transition(nxt, act2)
#             #     if nxt2 is None:
#             #         continue
#             #     cur_score = max(cur_score, 0.760666 * (cleared + cleared2) + eval_board(nxt2))
#             if cur_score > best_score:
#                 best_score, best_act = cur_score, act
#         moves += 1
#         state, cleared, done = transition(state, best_act)
#         if state is None:
#             break 

#         print_info(state)
#         if done:
#             print("Finished in", moves, "moves")
#             break