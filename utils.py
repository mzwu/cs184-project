import torch
import random

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

    # Swap held piece
    SWAP = (action > NUM_ACTIONS / 2)
    if SWAP:
        state[-8:-1] = state[-50:-43]
        state[-1] = 0 # im just gonna assume that this last slot equals 1 iff holding a piece

    # Update queue
    current_pieces = []
    for i in reversed(range(QUEUE_SIZE)):
        start = -((i + 3) * NUM_PIECES + 1)
        state[start:start+7] = state[start+7:start+14]
        # Grab current piece values
        arg = torch.argmax(state[start:start+7])
        current_pieces.append[arg]
    # Add next piece to queue - not counting current piece bc that would make it deterministic
    remainder = list(set(range(6)) - set(current_pieces))
    next_indicator = random.choice(remainder)
    next_piece = torch.zeros(NUM_PIECES, dtype=torch.int32)
    next_piece[next_indicator] = 1
    state[-15:-8] = next_piece

    return state


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
    total_indicators = row_length * num_rows + my_pieces
    total_cells = row_length * num_rows
    starting_holes = row_length * (num_rows - starting_rows)

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

    # Populate starting pieces (excluding held piece)
    for i in range(my_pieces - 1):
        # Start with all zeros
        row = torch.zeros(num_pieces, dtype=torch.int32)

        # Indicate which piece and add row
        row[sample[i]] = 1
        state[total_cells + i * num_pieces:total_cells + (i + 1) * num_pieces] = row 

    # Handle held piece
    # i think you can just make the last index one at the start to represent nothing held
    random_index = torch.randint(0, num_pieces + 1, (1,)).item()
    held_piece = torch.zeros(num_pieces + 1, dtype=torch.int32)
    held_piece[random_index] = 1
    state[- (num_pieces + 1):] = held_piece
    
    return state