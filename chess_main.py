import pandas as pd
import math

def generate_alphabetical_sequence():
    # Yield single characters first
    for char in 'abcdefghijklmnopqrstuvwxyz':
        yield char

    # Now for double characters
    for first_char in 'abcdefghijklmnopqrstuvwxyz':
        for second_char in 'abcdefghijklmnopqrstuvwxyz':
            yield first_char + second_char

# Function to get a list of x length
def get_sequence(length):
    gen = generate_alphabetical_sequence()
    return [next(gen) for _ in range(length)]

def get_number(s):
    # Define the mapping of letters to numbers
    letter_to_num = {char: idx+1 for idx, char in enumerate('abcdefghijklmnopqrstuvwxyz')}
    
    # Convert the sequence string to its corresponding number
    number = 0
    for idx, char in enumerate(reversed(s)):
        number += letter_to_num[char] * (26 ** idx)
    return number

class Board:

    def __init__(self, x, y):
        self.board = pd.DataFrame(columns=get_sequence(x), index=range(1,y+1))
        self.board = self.board.fillna(pd.NA)
        self.piece_locations = pd.DataFrame(columns=['x', 'y', 'piece'])
        self.history = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'piece', 'captured_piece', 'check', 'checkmate'])
    
    def __repr__(self):
        return str(self.piece_locations)

    def __str__(self):
        #first row of board_str = column names
        board_str = "  "
        for column in self.board.columns:
            board_str += f" {column} "
        board_str += "\n"
        #for each row in board, add row number and then each cell inside brackets
        for row in self.board.itertuples():
            board_str += f"{row[0]} "
            for cell in row[1:]:
                if pd.isna(cell):
                    board_str += "[ ]"
                else:
                    board_str += f"[{cell.symbol}]"
            board_str += "\n"
        return board_str
        
    def place_piece(self, piece, x, y):
        self.board.loc[y, x] = piece
        self.piece_locations = pd.concat([self.piece_locations, pd.DataFrame({'x': x, 'y': y, 'piece': piece}, index=[0])], ignore_index=True)
    
    def remove_piece(self, x, y):
        self.board.loc[y, x] = pd.NA
        self.piece_locations = self.piece_locations.loc[~((self.piece_locations['x'] == x) & (self.piece_locations['y'] == y))]
    
    def calculate_movement(self, x1, y1, x2, y2):
        if x1 == x2:
            if y1 == y2:
                return pd.NA
            return [0, y2 - y1, 0, 0]
        if y1 == y2:
            return [get_number(x2) - get_number(x1), 0, 0, 0]
        return [0, 0, (((y2 - y1) / abs(y2 - y1)) * (abs(get_number(x2) - get_number(x1)) + abs(y2 - y1))/2), (get_number(x2) - get_number(x1))/abs(get_number(x2) - get_number(x1))]
        
    def check_check(self, x, y, hypothetical_locations=False):
        """
        check if square can be attacked by any piece of the opposite color
        Keyword arguments:
        x -- x coordinate of square to check
        y -- y coordinate of square to check
        hypothetical_locations -- df to use instead of piece_locations (to check if move gets out of check)
        """
        #TODO: check only for pieces of opposite color to save time
        if isinstance(hypothetical_locations, pd.DataFrame):
            can_attack = hypothetical_locations.apply(lambda row: self.move_piece(row['x'], row['y'], x, y, row['piece'].color, kingcheck=True, do_move=False), axis=1)
        else:
            can_attack = self.piece_locations.apply(lambda row: self.move_piece(row['x'], row['y'], x, y, row['piece'].color, kingcheck=True, do_move=False), axis=1)
        if len(can_attack.loc[~can_attack.astype(str).str.startswith('ERR')]) > 0:
            return True
        return False
    
    def check_mate(self, x, y):
        """
        check if square can be attacked by any piece of the opposite color on its current square or on any square it can move to
        Keyword arguments:
        x -- x coordinate of square to check
        y -- y coordinate of square to check
        """
        if self.check_check(x, y):
            for i in self.board.columns:
                for j in self.board.index:
                    if self.move_piece(x, y, i, j, self.board.loc[y, x].color, kingcheck=True, do_move=False):
                        return False
            return True
        return False
        
    def check_stalemate(self, x, y):
        """
        check if square can be attacked by any piece of the opposite color on any square it can move to but not on its current square
        Keyword arguments:
        x -- x coordinate of square to check
        y -- y coordinate of square to check
        """
        if not self.check_check(x, y):
            for i in self.board.columns:
                for j in self.board.index:
                    if self.move_piece(x, y, i, j, self.board.loc[y, x].color, kingcheck=True, do_move=False):
                        return False
            #check if any pieces of same color can move
            own_pieces = self.piece_locations.loc[self.piece_locations['piece'].apply(lambda x: x.color) == self.board.loc[y, x].color]
            for piece in own_pieces.itertuples():
                for i in self.board.columns:
                    for j in self.board.index:
                        if self.move_piece(piece.x, piece.y, i, j, self.board.loc[piece.y, piece.x].color, kingcheck=True, do_move=False):
                            return False
            return True
        return False

    def move_piece(self, x1, y1, x2, y2, color, kingcheck=False, do_move=True):
        """
        check if you can move piece from x1, y1 to x2, y2
        Keyword arguments:
        x1 -- x coordinate of piece to move
        y1 -- y coordinate of piece to move
        x2 -- x coordinate of destination
        y2 -- y coordinate of destination
        color -- player color
        kingcheck -- whether to check if king is in check after the move (loop prevention)
        do_move -- whether to actually move the piece or just check if it is possible
        """
        capturing = False
        en_passant = False
        #check if piece on tile
        if pd.isna(self.board.loc[y1, x1]):
            return 'ERR: no piece!'
        #check if piece is of player color
        if self.board.loc[y1, x1].color != color:
            return 'ERR: not your piece!'
        #check if own king in check and if move is not to get out of check
        own_king = self.piece_locations.loc[(self.piece_locations['piece'].apply(lambda x: x.name) == 'king') & (self.piece_locations['piece'].apply(lambda x: x.color) == color)]
        if not kingcheck:
            if self.check_check(own_king['x'].iloc[0], own_king['y'].iloc[0]):
                hypothetical_locations = self.piece_locations.copy()
                hypothetical_locations = hypothetical_locations.loc[~((hypothetical_locations['x'] == x2) & (hypothetical_locations['y'] == y2))]
                hypothetical_locations.loc[(hypothetical_locations['x'] == x1) & (hypothetical_locations['y'] == y1), 'x'] = x2
                hypothetical_locations.loc[(hypothetical_locations['x'] == x1) & (hypothetical_locations['y'] == y1), 'y'] = y2
                if self.check_check(own_king['x'].iloc[0], own_king['y'].iloc[0], hypothetical_locations):
                    return 'ERR: king in check!'
        #check if own piece on destination tile
        if not pd.isna(self.board.loc[y2, x2]):
            if self.board.loc[y2, x2].color == self.board.loc[y1, x1].color:
                return 'ERR: own piece on tile!'
            capturing = True
        #check if en passant
        #first, check if moving piece can en passant
        if self.board.loc[y1, x1].movement.passant:
            #then, check history to see if last move was a pawn moving two squares
            if self.history.shape[0] > 0:
                if self.history.iloc[-1]['piece'].name == 'pawn':
                    if abs(self.calculate_movement(self.history.iloc[-1]['x1'], self.history.iloc[-1]['y1'], self.history.iloc[-1]['x2'], self.history.iloc[-1]['y2'])[1]) == 2:
                        #then, check if this move is to the space in between the opponent's pawn's current and previous position
                        if x2 == self.history.iloc[-1]['x2'] and y2 == self.history.iloc[-1]['y2'] - (self.history.iloc[-1]['y2'] - self.history.iloc[-1]['y1']) / 2:
                            en_passant = True
                            capturing = True
        #check if movement within piece movement
        movement = self.calculate_movement(x1, y1, x2, y2)
        #if capturing, check if capture movement allowed
        if capturing:
            allowed_movement = self.board.loc[y1, x1].movement.capture
        #else check if first move by looking in self.history for the piece and its position, and if it is not there, check if first move movement allowed
        elif self.history.loc[(self.history['piece'] == self.board.loc[y1, x1]) & (self.history['x2'] == x1) & (self.history['y2'] == y1)].shape[0] == 0:
            allowed_movement = self.board.loc[y1, x1].movement.first_move
        else:
            allowed_movement = self.board.loc[y1, x1].movement.movement
        allowed_movement.append(2.0)
        if movement > allowed_movement:
            return 'ERR: movement not allowed!'
        if movement == pd.NA:
            return 'ERR: movement not allowed!'
        #check knight and diagonal movement alignment
        if isinstance(movement[2], float):
            if not math.isclose(abs(movement[2]), allowed_movement[2]) and not math.isinf(allowed_movement[2]):
                return 'ERR: movement unaligned!'
            elif not math.isclose(((abs(get_number(x2) - get_number(x1)) / abs(y2 - y1))), 1.0):
                return 'ERR: movement unaligned!'
        #check if piece in way if hop not allowed
        if not self.board.loc[y1, x1].movement.hop:
            #check in x direction
            if movement[0] != 0:
                for i in range(1, abs(movement[0])):
                    if not pd.isna(self.board.iloc[y1 - 1, int(get_number(x1) - 1 + i * (movement[0] / abs(movement[0])))]):
                        return 'ERR: piece in the way!'
            #check in y direction
            if movement[1] != 0:
                for i in range(1, abs(movement[1])):
                    if not pd.isna(self.board.loc[int(y1 + i * (movement[1] / abs(movement[1]))), x1]):
                        return 'ERR: piece in the way!' 
            #check if piece in way if diagonal movement
            if movement[2] != 0:
                for i in range(1, int(abs(movement[2]))):
                    if not pd.isna(self.board.iloc[int(y1 - 1 + i * (movement[2] / abs(movement[2]))), int(get_number(x1) - 1 + i * movement[3])]):
                        return 'ERR: piece in the way!'
        #check if movement is backward if backward movement not allowed
        if not self.board.loc[y1, x1].movement.backward:
            if self.board.loc[y1, x1].color == 'white':
                if movement[0] < 0:
                    return 'ERR: backward movement not allowed!'
            else:
                if movement[0] > 0:
                    return 'ERR: backward movement not allowed!'
        #prevent king from moving into check
        if self.board.loc[y1, x1].movement.king:
            if not kingcheck:
                if self.check_check(x2, y2):
                    return 'ERR: king in check!'

        '''if self.board.loc[y1, x1].movement.passant:
            if movement[0] != 0:
                if pd.isna(self.board.loc[y2, x2]):
                    if not pd.isna(self.board.iloc[y1+1, get_number(x2)-1]) or not pd.isna(self.board.iloc[y1+1, get_number(x2)+1]):
                        if self.board.loc[y1, x2].name == 'pawn':
                            if self.board.loc[y1, x2].color != self.board.loc[y1, x1].color:
                                if self.board.loc[y1, x2].passant:
                                    return 'EN PASSANT'''        
        #upgrade pawn if possible
        if self.board.loc[y1, x1].name == 'pawn':
            if self.board.loc[y1, x1].movement.upgradable:
                if self.board.loc[y1, x1].color == 'white':
                    if y2 == 8:
                        self.board.loc[y1, x1] = self.board.loc[y1, x1].movement.upgradable
                else:
                    if y2 == 1:
                        self.board.loc[y1, x1] = self.board.loc[y1, x1].movement.upgradable
        #move piece
        if do_move:
            #TODO: check if opponent in check or checkmate to add to history
            self.opponent_king = self.piece_locations.loc[(self.piece_locations['piece'].apply(lambda x: x.name) == 'king') & (self.piece_locations['piece'].apply(lambda x: x.color) != color)]
            if en_passant:
                captured_piece = self.board.loc[y1, x2]
            else:
                captured_piece = self.board.loc[y2, x2]
            if self.check_check(self.opponent_king['x'].iloc[0], self.opponent_king['y'].iloc[0]):
                if self.check_mate(self.opponent_king['x'].iloc[0], self.opponent_king['y'].iloc[0]):
                    self.history = pd.concat([self.history, pd.DataFrame({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'piece': self.board.loc[y1, x1], 'captured_piece': captured_piece, 'check': True, 'checkmate': True}, index=[0])], ignore_index=True)
                    #self.history = self.history.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'piece': self.board.loc[y1, x1], 'captured_piece': captured_piece, 'check': True, 'checkmate': True}, ignore_index=True)
                else:
                    self.history = pd.concat([self.history, pd.DataFrame({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'piece': self.board.loc[y1, x1], 'captured_piece': captured_piece, 'check': True, 'checkmate': False}, index=[0])], ignore_index=True)
                    #self.history = self.history.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'piece': self.board.loc[y1, x1], 'captured_piece': captured_piece, 'check': True, 'checkmate': False}, ignore_index=True)
            else:
                self.history = pd.concat([self.history, pd.DataFrame({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'piece': self.board.loc[y1, x1], 'captured_piece': captured_piece, 'check': False, 'checkmate': False}, index=[0])], ignore_index=True)
                #self.history = self.history.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'piece': self.board.loc[y1, x1], 'captured_piece': captured_piece, 'check': False, 'checkmate': False}, ignore_index=True)
            if en_passant:
                self.remove_piece(x2, y1)
            self.place_piece(self.board.loc[y1, x1], x2, y2)
            self.remove_piece(x1, y1)
        return True
    
    def get_winner(self):
        if len(self.history) == 0:
            return False
        if self.history.iloc[-1]['checkmate']:
            return self.history.iloc[-1]['piece'].color
        try:
            if self.history.iloc[-1] == self.history.iloc[-5] == self.history.iloc[-9] and self.history.iloc[-2] == self.history.iloc[-6] == self.history.iloc[-10]:
                return 'draw'
        except:
            pass
        if self.check_stalemate(self.opponent_king['x'].iloc[0], self.opponent_king['y'].iloc[0]):
            return 'draw'
        try:
            #fifty move rule
            past_fifty_moves = self.history.iloc[-1] == self.history.iloc[-51]
            if past_fifty_moves.loc[past_fifty_moves['captured_piece'] == pd.NA].shape[0] == 50:
                if past_fifty_moves.loc[past_fifty_moves['piece'].name == 'pawn'].shape[0] == 0:
                    return 'draw'
        except:
            pass
        return False
        
class PieceMovement:

    def __init__(self, color, x, y, d, hop=False, backward=True, passant=False, cap_x=False, cap_y=False, cap_d=False, king=False, upgradable=False, first_x=False, first_y=False, first_d=False):
        """
        Initialize piece movement definition
        Keyword arguments:
        color -- color of the piece
        x -- movement in x direction
        y -- movement in y direction
        d -- movement in diagonal direction
        hop -- whether the piece can hop over other pieces
        backward -- whether the piece can move backward
        passant -- whether the piece can en passant
        cap_x -- capture movement in x direction, if not specified, x is used
        cap_y -- capture movement in y direction, if not specified, y is used
        cap_d -- capture movement in diagonal direction, if not specified, d is used
        king -- whether the piece is a king
        upgradable -- piece that this piece can be upgraded to
        first_x -- movement in x direction on first move
        first_y -- movement in y direction on first move
        first_d -- movement in diagonal direction on first move
        """
        self.color = color
        self.movement = [x,y,d]
        self.hop = hop
        self.backward = backward
        self.passant = passant
        if any([cap_x,cap_y,cap_d]):
            self.capture = [cap_x,cap_y,cap_d]
        else:
            self.capture = [x,y,d]
        self.king = king
        self.upgradable = upgradable
        if any([first_x,first_y,first_d]):
            self.first_move = [first_x, first_y, first_d]
        else:
            self.first_move = [x,y,d]

class Piece:

    def __init__(self, name, color, symbol):
        self.name = name
        self.color = color
        self.symbol = symbol
    
    def assign_movement(self, x, y, d, **kwargs):
        self.movement = PieceMovement(self.color, x, y, d, **kwargs)
    

#define default chess pieces along with their properties
pieces = {
    'pawn': {
        'x': 0,
        'y': 1,
        'd': 0,
        'backward': False,
        'passant': True,
        'cap_x': 0,
        'cap_y': 0,
        'cap_d': 1,
        'first_x': 0,
        'first_y': 2,
        'first_d': 0,
        'upgradable': True
    },
    'knight': {
        'x': 0,
        'y': 0,
        'd': 1.5,
        'hop': True
    },
    'bishop': {
        'x': 0,
        'y': 0,
        'd': math.inf,
    },
    'rook': {
        'x': math.inf,
        'y': math.inf,
        'd': 0,
    },
    'queen': {
        'x': math.inf,
        'y': math.inf,
        'd': math.inf,
    },
    'king': {
        'x': 1,
        'y': 1,
        'd': 1,
        'king': True
    }
}

pawn_white = Piece('pawn', 'white', '♙')
pawn_white.assign_movement(**pieces['pawn'])
pawn_black = Piece('pawn', 'black', '♟')
pawn_black.assign_movement(**pieces['pawn'])
knight_white = Piece('knight', 'white', '♘')
knight_white.assign_movement(**pieces['knight'])
knight_black = Piece('knight', 'black', '♞')
knight_black.assign_movement(**pieces['knight'])
bishop_white = Piece('bishop', 'white', '♗')
bishop_white.assign_movement(**pieces['bishop'])
bishop_black = Piece('bishop', 'black', '♝')
bishop_black.assign_movement(**pieces['bishop'])
rook_white = Piece('rook', 'white', '♖')
rook_white.assign_movement(**pieces['rook'])
rook_black = Piece('rook', 'black', '♜')
rook_black.assign_movement(**pieces['rook'])
queen_white = Piece('queen', 'white', '♕')
queen_white.assign_movement(**pieces['queen'])
queen_black = Piece('queen', 'black', '♛')
queen_black.assign_movement(**pieces['queen'])
king_white = Piece('king', 'white', '♔')
king_white.assign_movement(**pieces['king'])
king_black = Piece('king', 'black', '♚')
king_black.assign_movement(**pieces['king'])



#define default chess board piece positions
positions = [
    ('a', 1, rook_white),
    ('b', 1, knight_white),
    ('c', 1, bishop_white),
    ('d', 1, queen_white),
    ('e', 1, king_white),
    ('f', 1, bishop_white),
    ('g', 1, knight_white),
    ('h', 1, rook_white),
    ('a', 2, pawn_white),
    ('b', 2, pawn_white),
    ('c', 2, pawn_white),
    ('d', 2, pawn_white),
    ('e', 2, pawn_white),
    ('f', 2, pawn_white),
    ('g', 2, pawn_white),
    ('h', 2, pawn_white),
    ('a', 8, rook_black),
    ('b', 8, knight_black),
    ('c', 8, bishop_black),
    ('d', 8, queen_black),
    ('e', 8, king_black),
    ('f', 8, bishop_black),
    ('g', 8, knight_black),
    ('h', 8, rook_black),
    ('a', 7, pawn_black),
    ('b', 7, pawn_black),
    ('c', 7, pawn_black),
    ('d', 7, pawn_black),
    ('e', 7, pawn_black),
    ('f', 7, pawn_black),
    ('g', 7, pawn_black),
    ('h', 7, pawn_black)
]

#initialize board

board = Board(8,8)

for position in positions:
    board.place_piece(position[2], position[0], position[1])

#game loop
color = 'white'
while not board.get_winner():
    print(board)
    move = input("Enter move: ")
    if move == 'history':
        print(board.history)
        continue
    if move == 'pieces':
        print(board.piece_locations)
        continue
    if move == 'board':
        print(board.board)
        continue
    if move == 'exit':
        break
    if len(move) != 4:
        print('ERR: invalid move!')
        continue
    if not move[0] in board.board.columns or not move[2] in board.board.columns:
        print('ERR: invalid move!')
        continue
    if not int(move[1]) in board.board.index or not int(move[3]) in board.board.index:
        print('ERR: invalid move!')
        continue
    this_move = board.move_piece(move[0], int(move[1]), move[2], int(move[3]), color)
    if str(this_move).startswith('ERR'):
        print(this_move)
        continue
    if color == 'white':
        color = 'black'
    else:
        color = 'white'

if board.get_winner():
    print(board)
    print(f"winner: {board.get_winner()}")
        
