import random

def print_board(board):
    for i in range(3):
        print(f" {board[i*3]} | {board[i*3+1]} | {board[i*3+2]} ")
        if i < 2:
            print("-----------")

def check_winner(board):
    # Check rows
    for i in range(0, 9, 3):
        if board[i] == board[i+1] == board[i+2] != " ":
            return board[i]
    
    # Check columns
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] != " ":
            return board[i]
    
    # Check diagonals
    if board[0] == board[4] == board[8] != " ":
        return board[0]
    if board[2] == board[4] == board[6] != " ":
        return board[2]
    
    # Check for tie
    if " " not in board:
        return "Tie"
    
    return None

def minimax(board, depth, is_maximizing):
    result = check_winner(board)
    if result is not None:
        if result == "X":
            return -10 + depth
        elif result == "O":
            return 10 - depth
        else:
            return 0
    
    if is_maximizing:
        best_score = -float('inf')
        for i in range(9):
            if board[i] == " ":
                board[i] = "O"
                score = minimax(board, depth + 1, False)
                board[i] = " "
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i] == " ":
                board[i] = "X"
                score = minimax(board, depth + 1, True)
                board[i] = " "
                best_score = min(score, best_score)
        return best_score

def ai_move(board):
    best_score = -float('inf')
    best_move = None
    
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(board, 0, False)
            board[i] = " "
            if score > best_score:
                best_score = score
                best_move = i
    
    # If no best move (shouldn't happen in a valid game state), choose random
    if best_move is None:
        empty_spots = [i for i, spot in enumerate(board) if spot == " "]
        return random.choice(empty_spots)
    
    return best_move

def play_game():
    board = [" "] * 9
    current_player = "X"  # Human starts first
    
    while True:
        print_board(board)
        
        if current_player == "X":
            print("Your turn (X)")
            while True:
                try:
                    move = int(input("Enter position (1-9): ")) - 1
                    if 0 <= move <= 8 and board[move] == " ":
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number between 1 and 9.")
            board[move] = "X"
        else:
            
            print("AI's turn (O)")
            move = ai_move(board)
            board[move] = "O"
            print(f"AI chooses position {move + 1}")
        
        winner = check_winner(board)
        if winner:
            print_board(board)
            if winner == "Tie":
                print("It's a tie!")
            else:
                print(f"{winner} wins!")
            break
        
        current_player = "O" if current_player == "X" else "X"

if __name__ == "__main__":
    play_game()