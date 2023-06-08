from typing import List

from faker import Faker
from faker.providers import credit_card, phone_number, address

from gen.enums import TokensToBeChanged
from gen.raw_trades import RawTrade


class FullTrade(object):

    def __init__(self,
                 raw_trade,
                 cpty_account,
                 our_account,
                 cpty_agent,
                 our_agent,
                 cpty_contact_person,
                 our_contact_person,
                 cpty_address,
                 our_address):
        # The Raw Trade
        self.raw_trade = raw_trade

        # Enrichment
        self.cpty_account = cpty_account
        self.our_account = our_account
        self.cpty_agent = cpty_agent
        self.our_agent = our_agent
        self.cpty_contact_person = cpty_contact_person
        self.our_contact_person = our_contact_person
        self.cpty_address = cpty_address
        self.our_address = our_address

        # Possible Errors Introduced
        self.errors = dict()

        # Final Confirmation
        self.confirmation = None

    def save_trade_and_confirmation(self, file):

        with open(file + "_trade.txt", "w") as f:
            f.write(str(self.raw_trade))

        with open(file + "_trade_confirmation.txt", "w") as f:
            f.write(self.confirmation)

    def create_confirmation_from_template(self):
        ### TO CHANGE
        self.confirmation = self._fill_out_template(self.raw_trade.cpty.template)

    def _fill_out_template(self, template):

        tokens = [token.value for token in TokensToBeChanged]

        raw_trade_values = [str(self.raw_trade.cpty),
                  self.raw_trade.asset,
                  str(self.raw_trade.units),
                  str(self.raw_trade.price),
                  self.raw_trade.ccy,
                  self.raw_trade.direction,
                  str(self.raw_trade.trade_date),
                  str(self.raw_trade.settlement_date)]

        enrichment_values = [
            self.cpty_account,
            self.our_account,
            self.cpty_agent,
            self.our_agent,
            self.cpty_contact_person,
            self.our_contact_person,
            self.cpty_address,
            self.our_address
        ]

        filled_template = template

        possible_errors = self.errors

        for token, value in zip(tokens, raw_trade_values + enrichment_values):

            if token in possible_errors:
                error_value = str(possible_errors[token])
                filled_template = filled_template.replace(token, error_value)
            else:
                filled_template = filled_template.replace(token, value)

        return filled_template


def convert_raw_trades_to_full_trades(data_set: List[RawTrade]):
    full_trades = []
    for t in data_set:
        full_trades.append(enrich(t))

    return full_trades


def enrich(trade: RawTrade) -> FullTrade:

    fake = Faker()
    fake.add_provider(phone_number)
    fake.add_provider(address)

    cpty_account = fake.phone_number()
    our_account = fake.phone_number()
    cpty_agent = fake.name()
    our_agent = fake.name()
    cpty_contact_person = fake.name()
    our_contact_person = fake.name()
    cpty_address = fake.address()
    our_address = fake.address()

    return FullTrade(trade,
                     cpty_account,
                     our_account,
                     cpty_agent,
                     our_agent,
                     cpty_contact_person,
                     our_contact_person,
                     cpty_address,
                     our_address)
