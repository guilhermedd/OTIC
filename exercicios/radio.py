lucro_a = 180
lucro_b = 300

individuos = [
    "011011010100",
    "111100100100"
]

def get_fen_a(individuo):
    individuo_a = individuo[0:len(individuo) // 2]
    individuo_a = int(individuo_a[::-1], 2)
    fen = (individuo_a / 300) * 50
    return fen

def get_fen_b(individuo):
    individuo_b = individuo[len(individuo) // 2:]
    val = int(individuo_b[::-1], 2)
    fen = (val / 180) * 60
    return fen
 
def lucro(individuo):
    lucro_a = get_fen_a(individuo) *  180
    lucro_b = get_fen_b(individuo) * 300
    return lucro_a + lucro_b

def fitness(individuo):
    individuo_a = individuo[0:len(individuo) // 2]
    individuo_a = int(individuo_a[::-1], 2)
    individuo_b = individuo[len(individuo) // 2:]
    individuo_b = int(individuo_b[::-1], 2)
    discount = 1 / (120 - individuo_a - 2 * individuo_b)
    return lucro(individuo) / (180 * 60 + 300 * 50) - discount
    
    
def main():
    for individuo in individuos:
        print(f"Indivíduo: {individuo}")
        print(f"  FEN A: {get_fen_a(individuo)}")
        print(f"  FEN B: {get_fen_b(individuo)}")
        print(f"  Lucro: {lucro(individuo)}")
        print(f"  Fitness: {fitness(individuo)}")

if __name__ == "__main__":
    main()