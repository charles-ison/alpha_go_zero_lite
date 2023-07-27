import game_boards
import monte_carlo_tree as mct

tic_tac_toe = game_boards.TicTacToe()

turn_count = 0
while tic_tac_toe.detect_winner() is False:
    tic_tac_toe.print_board()
    player_num = (turn_count % 2) + 1
    row, column = input("\nPlayer " + str(player_num) +" please enter move coordinates: ").split(",")
    row, column = int(row), int(column)
    if tic_tac_toe.is_valid_move(row, column):
        tic_tac_toe.board[row][column] = player_num
        turn_count += 1
    else:
        print("\nInvalid move Player"  + str(player_num) + ". Please try again.")

tic_tac_toe.print_board()
print("\nPlayer " + str(player_num) + " won!")

root = mct.Node("X", 0, 0, is_root=True)
sub_tree = mct.Node("O", 0, 1)
root.add_sub_tree(sub_tree)