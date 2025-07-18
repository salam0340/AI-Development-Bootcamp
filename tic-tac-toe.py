import math

# Initialize the Tic-Tac-Toe board
def initialize_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

# Print the board
def print_board(board):
    for row in board:
        print('|'.join(row))
        print('-' * 5)

# Check for a winner
def check_winner(board, player):
    # Check rows, columns, and diagonals for a win
    for row in board:
        if all([cell == player for cell in row]):
            return True
    
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
        return True
    
    return False

# Check if the board is full (a draw)
def is_full(board):
    return all([cell != ' ' for row in board for cell in row])

# Evaluate the board state for the minimax algorithm
def evaluate(board):
    if check_winner(board, 'O'):  # AI wins
        return 1
    elif check_winner(board, 'X'):  # Player wins
        return -1
    else:
        return 0  # No winner (yet)

# Minimax algorithm to calculate the best move
def minimax(board, depth, is_maximizing):
    score = evaluate(board)

    # Base cases: return the score if the game is over
    if score == 1:  # AI wins
        return score - depth  # Prioritize faster wins
    if score == -1:  # Player wins
        return score + depth  # Prioritize slower losses
    if is_full(board):  # Draw
        return 0

    if is_maximizing:  # AI's turn (maximizer)
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'  # AI makes a move
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ' '  # Undo the move
                    best_score = max(score, best_score)
        return best_score
    else:  # Player's turn (minimizer)
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'  # Player makes a move
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ' '  # Undo the move
                    best_score = min(score, best_score)
        return best_score

# Get the best move for the AI (Minimax)
def get_best_move(board):
    best_score = -math.inf
    best_move = None

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'  # AI makes a move
                score = minimax(board, 0, False)
                board[i][j] = ' '  # Undo the move
                if score > best_score:
                    best_score = score
                    best_move = (i, j)

    return best_move

# Function for the player's move
def player_move(board):
    while True:
        move = input("Enter your move (row and column numbers from 1-3, e.g., '1 1' for top-left): ")
        try:
            row, col = map(int, move.split())
            if board[row - 1][col - 1] == ' ':
                board[row - 1][col - 1] = 'X'
                break
            else:
                print("This cell is already occupied! Try again.")
        except (ValueError, IndexError):
            print("Invalid input! Please enter row and column numbers between 1 and 3.")

# Main game loop
def play_game():
    board = initialize_board()
    print("Tic-Tac-Toe: You are 'X' and the AI is 'O'.")
    print_board(board)

    while True:
        # Player's move
        player_move(board)
        print_board(board)

        # Check if player wins
        if check_winner(board, 'X'):
            print("Congratulations! You win!")
            break

        # Check for draw
        if is_full(board):
            print("It's a draw!")
            break

        # AI's move
        print("AI is making a move...")
        ai_move = get_best_move(board)
        board[ai_move[0]][ai_move[1]] = 'O'
        print_board(board)

        # Check if AI wins
        if check_winner(board, 'O'):
            print("AI wins! Better luck next time.")
            break

        # Check for draw
        if is_full(board):
            print("It's a draw!")
            break

# Run the game
if __name__ == "__main__":
    play_game()
