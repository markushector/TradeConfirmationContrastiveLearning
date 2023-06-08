import random
from typing import List
import yfinance as yf

from gen.enums import Assets, Directions, Currencies
from gen.counter_parties import CounterPartyGenerator, CounterParty


class RawTrade(object):
    def __init__(self, cpty, asset, units,
                 price, ccy, direction,
                 trade_date="2022-09-05", settlement_date="2022-09-07"):

        # BUYER, SELLER INSTEAD OF DIRECTION, CTPY?

        self.cpty: CounterParty = cpty
        self.asset = asset
        self.units = units
        self.price = price
        self.ccy = ccy
        self.direction = direction
        self.trade_date = trade_date
        self.settlement_date = settlement_date

    def __str__(self):

        """return "{}\n{}\n{}".format(self.asset,
                                   self.price,
                                   self.ccy)"""
        return "{}\n{}\n{}\n{}\n{}\n{}\n{}".format(self.asset,
                                                   self.units,
                                                   self.price,
                                                   self.ccy,
                                                   self.direction,
                                                   self.trade_date,
                                                   self.settlement_date
                                                   )



class RawTradeGenerator(object):

    def __init__(self, args):
        self.args = args

    def get_raw_trade_list_random(self) -> List[RawTrade]:
        raw_trades = []
        tickers = {asset.value: yf.Ticker(asset.value).history(period="5y") for asset in Assets}
        cptys = CounterPartyGenerator().generate(self.args.cptys)

        for _ in range(self.args.number):
            asset = random.choice([asset.value for asset in Assets])
            cpty = random.choice(cptys)
            direction = random.choice([direction.value for direction in Directions])
            df = tickers[asset]

            #sampled_data_point = df.sample()

            #date = sampled_data_point.index.date[0]
            date = str(df.index[0]).split(' ')[0]
            #date = '2022'

            trade_date = date
            #print(f"Price from yfinance: {sampled_data_point.Close}")
            #price = round(float(sampled_data_point.Close), 2)
            #print(f"Price from yfinance but converted to float: {price}")
            #units = random.randint(1, 10)

            ### Simpler dataset! ###
            #direction = str(Directions.BUY.value)
            price = round(random.randint(1, 2500) + random.random(), 2)
            ccy = random.choice([ccy.value for ccy in Currencies])
            direction = random.choice([dire.value for dire in Directions])
            #price = 5
            units = random.randint(1, 10000)

            raw_trade = RawTrade(cpty,
                                 asset,
                                 units,
                                 price,
                                 ccy,
                                 direction,
                                 trade_date=trade_date,
                                 settlement_date=date)

            raw_trades.append(raw_trade)

        return raw_trades