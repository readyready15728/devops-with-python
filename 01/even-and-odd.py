def even_and_odd():
    current = 'Odd'

    while True:
        if current == 'Odd':
            current = 'Even'
        else:
            current = 'Odd'

        yield current

g = even_and_odd()

print([next(g) for _ in range(20)])
