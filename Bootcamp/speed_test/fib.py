

def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

if __name__ == "__main__":
    n = 40
    print("{answer}\n".format(answer=fib(n)))
