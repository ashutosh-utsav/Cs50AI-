from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0

knowledge0 = And(
    Not(And(AKnight, AKnave)),  
    Or(AKnight, AKnave),         
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight, AKnave)))
)

# Puzzle 1

knowledge1 = And(
    Not(And(AKnight, AKnave)),  
    Or(AKnight, AKnave),        

    Not(And(BKnight, BKnave)), 
    Or(BKnight, BKnave),       

    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, Not(And(AKnave, BKnave)))
)

# Puzzle 2

knowledge2 = And(
    Not(And(AKnight, AKnave)), 
    Or(AKnight, AKnave),       

    Not(And(BKnight, BKnave)), 
    Or(BKnight, BKnave),        

    Implication(AKnight, And(AKnight, BKnight)),
    Implication(AKnave, Not(And(AKnave, BKnave))),

    Implication(BKnight, And(BKnight, AKnave)),
    Implication(BKnave, Not(And(BKnave, AKnight)))
)

# Puzzle 3

knowledge3 = And(
    Not(And(AKnight, AKnave)), 
    Or(AKnight, AKnave),        

    Not(And(BKnight, BKnave)),  
    Or(BKnight, BKnave),        

    Not(And(CKnight, CKnave)),  
    Or(CKnight, CKnave),        

    
    Or(
     
        And(
            Implication(AKnight, AKnight),
            Implication(AKnave, Not(AKnight))
        ),
        
    
        And(
            Implication(AKnight, AKnave),
            Implication(AKnave, Not(AKnave))
        )
    ),

    Not(And(
     
        And(
            Implication(AKnight, AKnight),
            Implication(AKnave, Not(AKnight))
        ),
        
  
        And(
            Implication(AKnight, AKnave),
            Implication(AKnave, Not(AKnave))
        )
    )),

   
    Implication(BKnight, And(
        Implication(AKnight, AKnave),
        Implication(AKnave, Not(AKnave))
    )),

    Implication(BKnave, Not(And(
        Implication(AKnight, AKnave),
        Implication(AKnave, Not(AKnave))
    ))),



    Implication(BKnight, CKnave),
    Implication(BKnave, Not(CKnave)),

 
    Implication(CKnight, AKnight),
    Implication(CKnave, Not(AKnight))
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()