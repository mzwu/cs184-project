import torch
import random

piece_names = "IJLOSTZ"
piece_dims = [4, 3, 3, 2, 3, 3, 3]
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
        ind = torch.argmax(state[200 + i*7 : 200 + (i+1)*7])
        if torch.sum(state[200 + i*7 : 200 + (i+1)*7]) > 1:
            print("alert", i, torch.sum(state[200 + i*7 : 200 + (i+1)*7]))
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

def transition(state, action):
    """
    Life is pleasant.
    Death is peaceful.
    It's the transition that's troublesome.
                             - Isaac Asimov 
    """
    NUM_PIECES = 7
    QUEUE_SIZE = 5
    NUM_ACTIONS = 80 # change this number later

    # drop piece first to update board, return if failed
    state = drop_piece(state, action)
    if state == None:
        return None

    # Swap held piece
    SWAP = (action >= NUM_ACTIONS / 2)
    if SWAP:
        tmp = state[-7:0]
        state[-7:0] = state[-49:-42]
        state[-49:-42] = tmp

    # Update queue
    current_pieces = []
    for i in reversed(range(QUEUE_SIZE)):
        start = -((i + 3) * NUM_PIECES)
        state[start:start+7] = state[start+7:start+14]
        # Grab current piece values
        arg = torch.argmax(state[start:start+7])
        current_pieces.append(arg)
    # Add next piece to queue - not counting current piece bc that would make it deterministic
    remainder = list(set(range(6)) - set(current_pieces))
    next_indicator = random.choice(remainder)
    next_piece = torch.zeros(NUM_PIECES, dtype=torch.int32)
    next_piece[next_indicator] = 1
    state[-14:-7] = next_piece

    return state

def drop_piece(state, action):
    # extract piece information
    ind = torch.argmax(state[200:207])
    if action >= 40:
        ind = torch.argmax(state[-7:0])

    dim = piece_dims[ind]
    shape = piece_shapes[ind]

    # extract column and rotation from action number
    col = -1 + (action % 10)
    rot = (action // 10) % 4

    for _iter in range(rot):
        new_shape = [[0]*dim for i in range(dim)]
        for x in range(dim):
            for y in range(dim):
                new_shape[y][dim-1-x] = shape[x][y]
        shape = new_shape

    print("Dropping Piece", piece_names[ind], "at column", col)
    
    # find bounding box
    box = [dim, -1, dim, -1]
    for x in range(dim):
        for y in range(dim):
            if shape[x][y] == 1:
                box[0] = min(box[0], x)
                box[1] = max(box[1], x)
                box[2] = min(box[2], y)
                box[3] = max(box[3], y)
    
    # bad column
    if col + box[2] < 0 or col + box[3] >= 10:
        return None 
    
    board = state[0:200]
    piece = torch.zeros(200, dtype = torch.int32)
    for x in range(dim):
        for y in range(dim):
            if shape[x][y] == 1:
                piece[(x - box[0]) * 10 + (col + y)] = 1 
    
    # intersects with top of board
    if torch.sum(torch.mul(piece, board)) > 0:
        return None
    
    # try to move piece down
    while True:
        # piece is on the ground
        if torch.sum(piece[190:200]) > 0:
            break
            
        # if moving piece down causes intersection
        new_piece = torch.zeros(200, dtype = torch.int32)
        new_piece[10:200] = piece[0:190]
        if torch.sum(torch.mul(new_piece, board)) > 0:
            break 
        
        piece = new_piece
    
    new_state = state.detach().clone()
    new_state[0:200] = torch.add(new_state[0:200], piece)
    return new_state

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
        torch.Tensor: A 1D tensor of size 250 meeting the above constraints.
    """
    # Initialize variables
    num_pieces = 7  # total number of possible tetris pieces
    my_pieces = next_pieces + 2  # add for current and held piece
    total_indicators = row_length * num_rows + my_pieces * 7
    total_cells = row_length * num_rows
    starting_holes = row_length * (num_rows - starting_rows)

    print(num_pieces, my_pieces, total_indicators, total_cells, starting_holes)

    # Create a tensor of size 250 initialized to zeros
    state = torch.zeros(total_indicators, dtype=torch.int32)
    
    # Populate bottom 10 rows
    prev_hole = -1
    for i in range(starting_rows):  
        # Start with all ones
        row = torch.ones(row_length, dtype=torch.int32)  

        # Determine index of hole in current row
        possible_values = torch.tensor([i for i in range(row_length) if i != prev_hole], dtype=torch.int32)
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
        row = torch.zeros(num_pieces, dtype=torch.int32)

        # Indicate which piece and add row
        row[sample[i]] = 1
        state[total_cells + i * num_pieces:total_cells + (i + 1) * num_pieces] = row 
    
    return state

if __name__ == "__main__":
    state = starting_position()
    print_info(state)
    while True:
        state = transition(state, random.randint(1, 7))
        if state == None:
            break 

        print_info(state)