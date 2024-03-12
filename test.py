def drop(num, depth):
    # print(f'--{num}--{depth}')
    row = pow(2, depth)
    while num > row:
        num -= row
        depth -= 1
        row = pow(2,depth)
    if num%2==0:
        now = row+row//2
        return now + num//2-1
    else:
        return row + num//2
    
size = int(input())
for i in range(size):
    depth, num = map(int, input().split())

    ans = drop(num, depth-1)
    print(ans)




# def sort(arr):
#     for i in range(len(arr)):
#         for j in range(i+1, len(arr)):
#             if(arr[i]>arr[j]):
#                 tmp = arr[i]
#                 arr[i] = arr[j]
#                 arr[j] = tmp
#     return arr

# class hash_table:
#     def __init__(self, size):
#         self.size = size
#         self.table = [[] for _ in range(size)]

#     def hash_func(self, key):
#         return key % self.size
    
#     def insert(self, key):
#         index = self.hash_func(key)
#         if key not in self.table[index]:
#             self.table[index].append(key)

#     def delete(self, key):
#         index = self.hash_func(key)
#         if key in self.table[index]:
#             self.table[index].remove(key)

#     def output_table(self):
#         # print(self.size)
#         for i in range(self.size):
#             lengh = len(self.table[i])
#             if lengh > 1:
#                 self.table[i] = sort(self.table[i])

#             if i < 10 and lengh==0: print(f'[00{i}]:NULL')
#             if i < 10 and lengh==0: print(f'[00{i}]: {self.table[i]}')
#             elif i < 100 : print(f'[0{i}]: {self.table[i]}')
#             else :  print(f'[{i}]: {self.table[i]}')


# k,m = map(int, input().split())
# table = hash_table(m)
# for i in range(k):
#     row = input().split()
#     if len(row)>1:
#         first, sec = int(row[0]), int(row[1])
#     elif len(row) == 1:
#         first = int(row[0])

#     if first==1:
#         table.insert(sec)
#         # table.output_table()

#     elif first==2:
#         table.delete(sec)
    
#     elif first==3:
#         print('==== s ====')
#         table.output_table()
#         print('==== e ====')


# N, M, K = map(int, input().split())
# flow =[]
# for i in range(N): flow.append([int(x) for x in input().split()])
# ans =[]

# for i in range(K):
#     router = [[]for _ in range(M)]
#     plan = [int(x) for x in input().split()]
#     for j in range(N): router[plan[j]].append(j)
#     # print(router)
#     output = 0
#     for s1 in range(M):
#         for s2 in range(M):
#             if s1 ==s2 :
#                 sum_flow=0
#                 if (router[s1]==[]): continue
#                 sum_flow+=sum(flow[i][s2] for i in router[s1])
#                 output += sum_flow
#             else:
#                 sum_flow=0
#                 if router[s1]==[]: continue
#                 sum_flow+= sum([flow[i][s2] for i in router[s1]])
#                 if sum_flow<=1000: output+=sum_flow*3
#                 elif sum_flow>1000: output+=3000+(1000-sum_flow)*2
#     ans.append(output)
# print(min(ans))

            

# print(array)
# N = input()
# numbers = N.split()
# lengh = len(numbers)
# for i in range(lengh): numbers[i] = int(numbers[i])
# # print(numbers)

# global ans, idx
# ans = 0
# def explore(idx, end):
#     global ans
#     if idx < lengh:
#         # print(idx, numbers[idx])
#         if numbers[idx]==0:
#             # ans +=  abs(numbers[idx-1]-numbers[idx-2])
#             # print(numbers[idx-1], numbers[idx-2], numbers[idx-1]-numbers[idx-2])
#             return idx, 0
#         elif numbers[idx]%2 == 0:
#             r1, fin1 = explore(idx+1, 0)
#             if fin1 == 1: 
#                 ans += abs(numbers[idx]-numbers[idx+1])
           
#             r2, fin2 = explore(r1+1, 0)
#             if fin2 == 1 : 
#                 ans+=abs(numbers[idx]-numbers[r1+1])
            
#             if numbers[r1] == 0 and numbers[r2] == 0:
#                 return r2, 1
#             else:
#                 return r2, 0
#         elif numbers[idx]%2 == 1:
#             r1, fin1 = explore(idx+1, 0)
#             if fin1 == 1: 
#                 ans += abs(numbers[idx]-numbers[idx+1])

#             r2, fin2 = explore(r1+1, 0)
#             if fin2 == 1: 
#                 ans += abs(numbers[idx]-numbers[r1+1])
            
#             r3, fin3 = explore(r2+1, 0)
#             if fin3 == 1: 
#                 ans += abs(numbers[idx]-numbers[r2+1])
#             if numbers[r1] == 0 and numbers[r2] == 0 and numbers[r3] == 0:
#                 return r3, 1
#             else:
#                 return r3, 0

# explore(0, 0)
# print(ans)

## stack
# N = int(input())
# for i in range(N):
#     my_list = []
#     a = input()
#     num_A, num_B = 0, 0
#     non = 0
#     for j in a:
#         lengh = len(my_list)
        
#         if(j=='('):
#             my_list.append(j) 
#         elif(j=='['):
#             my_list.append(j)
#         elif(j==')'):
#             if lengh >0:
#                 if(my_list[lengh-1]=='('):
#                     my_list.pop()
#                 else : my_list.append(j)
#             else : my_list.append(j)
#         elif(j==']'):
#             if lengh >0:
#                 if(my_list[lengh-1]=='['):
#                     my_list.pop()
#                 else : my_list.append(j)
#             else : my_list.append(j)
#         else:
#             non += 1
    
#     if len(my_list)==0: print('Yes')
#     else: print('No')


            

## gcd
# N = int(input())

# def gcd(a, b):
#     while b:
#       a,b = b, a%b
#     return a

# for i in range(N):
#   number = (input())
#   numbers = number.split(' ')
#   max = 0
#   lengh = len(numbers)
#   for j in range(lengh-1):
#     for k in range(j+1, lengh):
#         if j != k :
#             if (numbers[j] and numbers[k]):
#                 ans = gcd(int(numbers[j]), int(numbers[k]))
#                 if (ans > max): max = ans
#   print(max)
        