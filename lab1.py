import pprint


def isSafe(board, x, y, n):
    # Checking whether the column is filled
    for row in range(x):
        if board[row][y] == 'Q':
            return False

    # Checking for top left diagonals
    row = x
    col = y
    while row >= 0 and col >= 0:
        if board[row][col] == 'Q':
            return False
        row -= 1
        col -= 1

    # Checking for top right diagonals
    row = x
    col = y
    while row >= 0 and col < n:
        if board[row][col] == 'Q':
            return False
        row -= 1
        col += 1

    # Return True if all tests are passed
    return True


def nQueen(board, x, n):
    # If we have successfully placed n queens
    if x >= n:
        return True

    # Iterate through columns for each row
    for col in range(n):
        # If the position is safe, place the queen
        if isSafe(board, x, col, n):
            board[x][col] = 'Q'

            # If the next queen can be placed
            if nQueen(board, x + 1, n):
                return True

            # Backtrack
            board[x][col] = ' '

    return False


n = int(input("Enter number of Q: "))

board = [[' '] * n for i in range(n)]

if nQueen(board, 0, n):
    pprint.pprint(board)
else:
    print("No Solution")
    