Mutez = int


def __michelson_get_amount() -> Mutez:
    return 0


def get_amount():
    amount = __michelson_get_amount()
    return amount
