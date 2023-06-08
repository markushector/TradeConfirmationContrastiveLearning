import random
from typing import List
import random as rn

from gen.enums import Assets, Currencies, Directions, TokensToBeChanged
from gen.full_trades import FullTrade


def generate_errors(full_trades: List[FullTrade], args) -> List[FullTrade]:

    full_trades_with_errors = []
    error_prob: float = args.error_probability
    error_prob_ = 0
    for full_trade in full_trades:

        p = rn.random()
        if p <= error_prob_:
            print('Error generated')
            wrong_asset = rn.choice([asset.value for asset in Assets])
            full_trade.errors[TokensToBeChanged.ASSET.value] = wrong_asset

        p = rn.random()
        if p <= error_prob:
            print('Error generated')
            wrong_units = rn.randint(0, 500)
            full_trade.errors[TokensToBeChanged.UNITS.value] = wrong_units

        p = rn.random()
        if p <= error_prob:
            print('Error generated')
            wrong_price = rn.randint(0, 500) + rn.random()
            #wrong_price = full_trade.raw_trade.price + 0.3 * (random.random() - 0.5)
            full_trade.errors[TokensToBeChanged.PRICE.value] = round(wrong_price, 2)

        p = rn.random()
        if p <= error_prob_:
            print('Error generated')
            wrong_ccy = rn.choice([currency.value for currency in Currencies])
            full_trade.errors[TokensToBeChanged.CURRENCY.value] = wrong_ccy

        if p <= error_prob_:
            print('Error generated')
            current_direction = full_trade.raw_trade.direction
            wrong_direction = Directions.BUY.value if current_direction == Directions.SELL.value \
                else Directions.BUY.value
            full_trade.errors[TokensToBeChanged.DIRECTION.value] = wrong_direction

        # TRADE AND VALUE DATE ERRORS

        full_trades_with_errors.append(full_trade)
    return full_trades_with_errors
