from heapq import heappush,heappop

start = (
    (5,8,2),
    (1,0,3),
    (4,7,6)
)
goal = (
    (1,2,3),
    (4,5,6),
    (7,8,0)
)
#h1 : number of 
def h1(state):
    count=0
    for i in range (3):
        for j in range(3):
            if state [i][j]!=0 and state[i][j]!= goal[i][j]:
                count = count +1
    return count
def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state [i][j]==0:
                return i,j 
            
def get_neighbors(state):
    x,y =find_blank(state)
    moves = [(-1,0),(0,-1),(1,0),(0,1)]
    neighbors =[]
    for dx,dy in moves:
        nx = x + dx
        ny = y +dy
        if 0 <= nx < 3 and 0 <=ny < 3:
            board = [list(row) for row in state]
            board[x][y],board[nx][ny]= board[nx][ny],board[x][y]
            neighbors.append(tuple (map(tuple,board)))
    return neighbors
def dispaly(state):
    for row in state:
        print(row)
    print()
def aStar(start,heuristic):
    pq =[]
    heappush(pq,(heuristic(start),0,start))
    parent = {start: None}
    g_cost = {start: 0}
    expanded =0 
    while pq:
        f,g ,current = heappop(pq)
        expanded =expanded +1
        if current == goal:
            return True,  parent,expanded
        for neighbors in get_neighbors(current):
            naya_g =g +1
            if neighbors not in g_cost or naya_g < g_cost[neighbors]:
                g_cost[neighbors] = naya_g
                f_cost = naya_g + heuristic(neighbors)
                heappush(
                    pq, (f_cost,naya_g,neighbors)
                    
                )
                parent[neighbors]= current
    return False,parent,expanded 




def reconstruct(parent, current):
    path = []
    while current is not None:
        path.append(current)
        current = parent[current]
    return path[::-1]


goalfound, parent, expanded = aStar(start, h1)

if goalfound:
    print(f"Goal state found after expanding {expanded} states\n")

    for state in reconstruct(parent, goal):
        dispaly(state)
else:
    print(f"Goal state not found after expanding {expanded} states")
        