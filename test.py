
def gcd(a, b):
    print("called")
    if (a == b):
        return a

    if (a < b):
        return gcd(a, b-a)
    else:
        return gcd(a-b, b)





