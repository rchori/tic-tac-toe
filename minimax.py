theBoard = {0: ' ' , 1: ' ' , 2: ' ',
            3: ' ' , 4: ' ' , 5: ' ',
            6: ' ' , 7: ' ' , 8: ' ' }

board_keys = []

for key in theBoard:
    board_keys.append(key)

huPlayer = 'x'
aiPlayer = 'o'

def printBoard(board):
    print(board[0] + '|' + board[1] + '|' + board[2]),
    print('-+-+-'),
    print(board[3] + '|' + board[4] + '|' + board[5]),
    print('-+-+-'),
    print(board[6] + '|' + board[7] + '|' + board[8])
    
#printBoard(theBoard)

def get_possible_moves(board):
    possibleMoves = []
    for i in board:
        if board[i] == ' ':
            possibleMoves.append(i)
    return possibleMoves

#get_possible_moves(theBoard)

def winning(board, player):
    if any([theBoard[0] == theBoard[1] == theBoard[2] == player,
            theBoard[3] == theBoard[4] == theBoard[5] == player,
            theBoard[6] == theBoard[7] == theBoard[8] == player,
            theBoard[0] == theBoard[3] == theBoard[6] == player,
            theBoard[1] == theBoard[4] == theBoard[7] == player,
            theBoard[2] == theBoard[5] == theBoard[8] == player,
            theBoard[0] == theBoard[4] == theBoard[8] == player,
            theBoard[2] == theBoard[4] == theBoard[6] == player]):
        return True
    else:
        return False

def minimax(newBoard, player):
    
    availSpots = get_possible_moves(newBoard)
    
    if winning(newBoard, huPlayer):
        return {'score':-10}
    elif winning(newBoard, aiPlayer):
        return {'score':10}
    elif len(availSpots) == 0:
        return {'score':0}

    moves = []
    
    for i in range (0, len(availSpots)):
        move = {}
        move['index'] = availSpots[i]
        
        newBoard[availSpots[i]] = player
        
        if (player == aiPlayer):
            result = minimax(newBoard, huPlayer)
            move['score'] = result['score']
        else:
            result = minimax(newBoard, aiPlayer)
            move['score'] = result['score']
        
        newBoard[availSpots[i]] = ' '
        
        moves.append(move)
    
    if(player == aiPlayer):
        bestScore = -10000;
        for i in range (0, len(moves)):
            if(moves[i]['score'] > bestScore):
                bestScore = moves[i]['score']
                bestMove = i
    else:
        bestScore = 10000
        for i in range (0, len(moves)):
            if(moves[i]['score'] < bestScore):
                bestScore = moves[i]['score']
                bestMove = i
    
    return moves[bestMove]

def game():
    
    player = huPlayer
    count = 0
    
    while (count < 9):
        printBoard(theBoard)
        print(count)
        print("It's your turn," + player + ". Move to which cell?")
        
        if player == huPlayer:
            move = int(input())
            if theBoard[move] == ' ':
                theBoard[move] = player
                count += 1
            else:
                print('That cell is already filled!')
                continue
        else:
            moveAI = minimax(theBoard, player)['index']
            theBoard[moveAI] = player
            count += 1
        
        if count >= 5:
            if winning(theBoard, player):
                printBoard(theBoard)
                print("Game Over.")
                print(" **** " + player + " won. ****")
                break
        
        if player == huPlayer:
            player = aiPlayer
        else:
            player = huPlayer
    else:
        print("Game Over.")
        print("It's a Tie!!")
    
    restart = input("Do you want to play Again?(y/n)")
    if restart == 'y' or restart == 'Y':
        for key in board_keys:
            theBoard[key] = ' '
        
        game()

if __name__ == "__main__":
    game()
