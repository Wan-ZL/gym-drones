import pulp
import numpy as np

if __name__ == '__main__':
    MyProbLP = pulp.LpProblem("MD_Demo", sense=pulp.LpMinimize)

    E_T = 1
    map_size = 3
    E = {(0,0),(0,1),(1,0),(1,1)}
    #                   (i_x, i_y), (j_x, j_y)
    # d_ij = np.zeros((map_size,map_size,map_size,map_size))

    # Base station location (BS and cell locations cannot be the same)
    baseStation_x = map_size+1
    baseStation_y = map_size+1

    # distances
    d_ij = {}
    for i_x in range(map_size):
        for i_y in range(map_size):
            # cell to cell distance
            for j_x in range(map_size):
                for j_y in range(map_size):
                    if (i_x,i_y) != (j_x,j_y):
                        d_ij[(i_x, i_y), (j_x, j_y)] = np.linalg.norm([i_x - j_x, i_y - j_y])
            # base station to cell distances
            d_ij[(baseStation_x, baseStation_y), (i_x, i_y)] = np.linalg.norm([baseStation_x - i_x, baseStation_y - i_y])
    print(d_ij)

    # create variables (x_ijk)
    X_test_k= pulp.LpVariable('X_test_k', lowBound=0, upBound=1, cat='Integer')
    # variables for edge
    X_ijk = {}
    for i_x in range(map_size):
        for i_y in range(map_size):
            # for cells
            for j_x in range(map_size):
                for j_y in range(map_size):
                    if (i_x, i_y) != (j_x, j_y):
                        X_ijk[(i_x,i_y),(j_x,j_y)] = pulp.LpVariable(f'X_({i_x},{i_y})({j_x},{j_y})k', lowBound=0, upBound=1, cat='Integer')
            # for base station to cell
            X_ijk[(baseStation_x, baseStation_y), (i_x, i_y)] = pulp.LpVariable(f'X_({baseStation_x},{baseStation_y})({i_x},{i_y})k', lowBound=0, upBound=1, cat='Integer')
    print(type(X_ijk[(0,0),(1,1)]))
    print(X_ijk)

    # variables for cell (node)
    y_ik = {}   # non-negative integer
    # for cells
    for i_x in range(map_size):
        for i_y in range(map_size):
            y_ik[(i_x,i_y)] = pulp.LpVariable(f'y_({i_x},{i_y})k', lowBound=0, cat='Integer')
    # for base station
    y_ik[(baseStation_x, baseStation_y)] = pulp.LpVariable(f'y_({baseStation_x},{baseStation_y})k', lowBound=0, cat='Integer')


    # create target function
    sum_value = 0
    # MyProbLP += E_T * d_ij[0][0] * X_test_k
    all_variable = 0 #pulp.LpVariable('init', lowBound=0, upBound=0, cat='Integer')
    for i_x in range(map_size):
        for i_y in range(map_size):
            # cells to cells
            for j_x in range(map_size):
                for j_y in range(map_size):
                    if (i_x, i_y) != (j_x, j_y):
                        sum_value += E_T * d_ij[(i_x,i_y),(j_x,j_y)] * X_ijk[(i_x,i_y),(j_x,j_y)]
                        all_variable += X_ijk[(i_x,i_y),(j_x,j_y)]
            # base station to cells
            sum_value += E_T * d_ij[(baseStation_x, baseStation_y), (i_x, i_y)] * X_ijk[(baseStation_x, baseStation_y), (i_x, i_y)]
            # all_variable += X_ijk[(baseStation_x, baseStation_y), (i_x, i_y)]
    print("sum_value", sum_value)
    # save target function
    MyProbLP += sum_value


    # constrain functions
    # Eq.(5)
    t_i = 1
    for i_x in range(map_size):
        for i_y in range(map_size):
            MyProbLP += (y_ik[(i_x,i_y)] == t_i)
    # Eq.(7)
    for i_x in range(map_size):
        for i_y in range(map_size):
            for j_x in range(map_size):
                sum_X_ijk = 0
                for j_y in range(map_size):
                    if (i_x, i_y) != (j_x, j_y):
                        sum_X_ijk += X_ijk[(i_x,i_y),(j_x,j_y)]
            MyProbLP += (y_ik[(i_x,i_y)] <= t_i * sum_X_ijk)
    # Eq.(9)
    eq_9_sum = 0
    for j_x in range(map_size):
        for j_y in range(map_size):
            eq_9_sum += X_ijk[(baseStation_x, baseStation_y), (j_x, j_y)]
    MyProbLP += (eq_9_sum == 1)
    # Eq.(10)
    MyProbLP += (all_variable <= (map_size*map_size-1)) # Error: should be '>=' or '<='?


    MyProbLP.solve()
    print("MyProbLP.variables()",MyProbLP)
    for v in MyProbLP.variables():
        print(v.name, "=", v.varValue)
    print("F(x) = ", pulp.value(MyProbLP.objective))

    # X_ijk = pulp.LpVariable('X_ijk', lowBound=0, upBound=1, cat='Integer')
    # i_index = pulp.LpVariable('i_index', lowBound=0, upBound=1, cat='Integer')
    # j_index = pulp.LpVariable('j_index', lowBound=0, upBound=1, cat='Integer')
    #
    #
    # MyProbLP += E_T * i_index * X_ijk
    # MyProbLP.solve()
    # for v in MyProbLP.variables():
    #     print(v.name, "=", v.varValue)
    # print("F(x) = ", pulp.value(MyProbLP.objective))


    # MyProbLP = pulp.LpProblem("LPProbDemo1", sense=pulp.LpMaximize)

    # x1 = pulp.LpVariable('x1', lowBound=0, upBound=7, cat='Continuous')
    # x2 = pulp.LpVariable('x2', lowBound=0, upBound=7, cat='Continuous')
    # x3 = pulp.LpVariable('x3', lowBound=0, upBound=7, cat='Continuous')
    #
    # MyProbLP += 2 * x1 + 3 * x2 - 5 * x3
    #
    # MyProbLP += (2 * x1 - 5 * x2 + x3 >= 10)  # 不等式约束
    # MyProbLP += (x1 + 3 * x2 + x3 <= 12)  # 不等式约束
    # MyProbLP += (x1 + x2 + x3 == 7)  # 等式约束
    #
    # MyProbLP.solve()
    # print("Status:", pulp.LpStatus[MyProbLP.status])  # 输出求解状态
    # for v in MyProbLP.variables():
    #     print(v.name, "=", v.varValue)  # 输出每个变量的最优值
    # print("F(x) = ", pulp.value(MyProbLP.objective))  # 输出最优解的目标函数值
