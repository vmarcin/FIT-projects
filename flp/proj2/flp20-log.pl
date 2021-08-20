/*
 * project:    Rubic's Cube Solver
 * author:     MARCIN Vladimir (xmarci10@stud.fit.vutbr.cz) 
 */

splitRev(0,[]).
splitRev(N,[A|As]) :- N1 is floor(N/10), A is N mod 10, splitRev(N1,As).

split([],[]).
split([N|Tail],[L1|L3]) :- splitRev(N,L2), reverse(L1,L2), split(Tail,L3).

getCube(C) :- 
    readln(W, end_of_file,_,"123456",_),    % return's list of cube rows in triples (e.g 111)
    split(W, N),                            % split each triple into list of digits (e.g 111 to [1,1,1]) 
    flatten(N,C).                           % flatten the list of lists

% transform input into internal representation (list to term)
% U - Upward face, F - Front face, R - Right face, 
% L - Left face,  B - Back face, D - Downward face
tiles(
    [
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    ],
    cube(
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    ) 
).

% prints cube in the desired format
print_cube( cube(
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    )) :-
    format("~w~w~w\n~w~w~w\n~w~w~w\n", [U1, U2, U3, U4, U5, U6, U7, U8, U9]),
    format("~w~w~w ~w~w~w ~w~w~w ~w~w~w\n", [F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3]),
    format("~w~w~w ~w~w~w ~w~w~w ~w~w~w\n", [F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6]),
    format("~w~w~w ~w~w~w ~w~w~w ~w~w~w\n", [F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9]),
    format("~w~w~w\n~w~w~w\n~w~w~w\n", [D1, D2, D3, D4, D5, D6, D7, D8, D9]).

print_result(_,[]).
print_result(Cube, [Move|Tail]) :- 
    move(Move, Cube, C2), nl, print_cube(C2), print_result(C2, Tail).

% representation of solved cube
ghoul( 
    cube(
        U, U, U, 
        U, U, U,
        U, U, U,
        F, F, F, R, R, R, B, B, B, L, L, L,
        F, F, F, R, R, R, B, B, B, L, L, L,
        F, F, F, R, R, R, B, B, B, L, L, L,
        D, D, D,
        D, D, D,
        D, D, D
    )
).

% rotate upward face clockwise
mov(u, 
    cube(
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    ),
    cube(
        U7, U4, U1,
        U8, U5, U2, 
        U9, U6, U3, 
        R1, R2, R3, B1, B2, B3, L1, L2, L3, F1, F2, F3, 
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    )
).

% rotate downward face clockwise
mov(d, 
    cube(
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    ),
    cube(
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        L7, L8, L9, F7, F8, F9, R7, R8, R9, B7, B8, B9, 
        D7, D4, D1, 
        D8, D5, D2,  
        D9, D6, D3  
    )
).

% rotate front face clockwise
mov(f, 
    cube(
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    ),
    cube(
        U1, U2, U3, 
        U4, U5, U6,
        L9, L6, L3,
        F7, F4, F1, U7, R2, R3, B1, B2, B3, L1, L2, D1,
        F8, F5, F2, U8, R5, R6, B4, B5, B6, L4, L5, D2,
        F9, F6, F3, U9, R8, R9, B7, B8, B9, L7, L8, D3,
        R7, R4, R1,
        D4, D5, D6,
        D7, D8, D9
    )
).

% rotate back face clockwise
mov(b, 
    cube(
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    ),
    cube(
        R3, R6, R9, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, D9, B7, B4, B1, U3, L2, L3,
        F4, F5, F6, R4, R5, D8, B8, B5, B2, U2, L5, L6,
        F7, F8, F9, R7, R8, D7, B9, B6, B3, U1, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        L1, L4, L7
    )
).

% rotate left face clockwise
mov(l, 
    cube(
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    ),
    cube(
        B9, U2, U3, 
        B6, U5, U6,
        B3, U8, U9,
        U1, F2, F3, R1, R2, R3, B1, B2, D7, L7, L4, L1,
        U4, F5, F6, R4, R5, R6, B4, B5, D4, L8, L5, L2,
        U7, F8, F9, R7, R8, R9, B7, B8, D1, L9, L6, L3,
        F1, D2, D3,
        F4, D5, D6,
        F7, D8, D9
    )
).

% rotate right face clockwise
mov(r, 
    cube(
        U1, U2, U3, 
        U4, U5, U6,
        U7, U8, U9,
        F1, F2, F3, R1, R2, R3, B1, B2, B3, L1, L2, L3,
        F4, F5, F6, R4, R5, R6, B4, B5, B6, L4, L5, L6,
        F7, F8, F9, R7, R8, R9, B7, B8, B9, L7, L8, L9,
        D1, D2, D3,
        D4, D5, D6,
        D7, D8, D9
    ),
    cube(
        U1, U2, F3, 
        U4, U5, F6,
        U7, U8, F9,
        F1, F2, D3, R7, R4, R1, U9, B2, B3, L1, L2, L3,
        F4, F5, D6, R8, R5, R2, U6, B5, B6, L4, L5, L6,
        F7, F8, D9, R9, R6, R3, U3, B8, B9, L7, L8, L9,
        D1, D2, B7,
        D4, D5, B4,
        D7, D8, B1
    )
).

% clockwise
move(+M, Old, New) :- mov(M, Old, New).
% counterclockwise
move(-M, Old, New) :- mov(M, New, Old).

% try rotations
rot(Cube, [], Cube).
rot(Cube, [Move|Tail], TransformedCube) :- 
    rot(RotatedCube, Tail, TransformedCube), move(Move, Cube, RotatedCube).

% solve cube by trying possible rotations
% 'Solution' represents list of moves to solve the 'Cube'
solve(Cube, Solution) :- rot(Cube, Solution, TransformedCube), ghoul(TransformedCube).

start :-
    prompt(_, ''), 
    getCube(Cube), tiles(Cube, Tiles), print_cube(Tiles),
    solve(Tiles, Solution), print_result(Tiles, Solution),
    halt.