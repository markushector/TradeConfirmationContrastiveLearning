import enum


class Assets_(enum.Enum):
    TSLA = 'TSLA'
    AAPL = 'AAPL'
    MSFT = 'MSFT'
    AMZN = 'AMZN'
    UNH = 'UNH'
    JNJ = 'JNJ'

class Assets(enum.Enum):
    TSLA = 'TSLA'
    AAPL = 'AAPL'
    MSFT = 'MSFT'
    AMZN = 'AMZN'
    GOOGL = 'GOOGL'
    GOOG = 'GOOG'
    UNH = 'UNH'
    JNJ = 'JNJ'
    XOM = 'XOM'
    JPM = 'JPM'

class Directions_(enum.Enum):
    BUY = 'wffwa'
    SELL = 'vafvfvs'

class Directions(enum.Enum):
    BUY = 'BUY'
    SELL = 'SELL'

class Currencies_(enum.Enum):
    SEK = 'sfdg'
    EUR = 'sdhg'
    DKK = 'wffgeg'
    USD = 'tehtwhht'
    ISK = 'bterb'
    CHF = 'gmjggj'

class Currencies(enum.Enum):
    SEK = 'SEK'
    EUR = 'EUR'
    DKK = 'DKK'
    USD = 'USD'
    ISK = 'ISK'
    CHF = 'CHF'


class TokensToBeChanged(enum.Enum):
    # Raw Trade Info
    COUNTERPARTY = 'COUNTERPARTY'
    ASSET = 'ASSET'
    UNITS = 'UNITS'
    PRICE = 'PRICE'
    CURRENCY = 'CURRENCY'
    DIRECTION = 'DIRECTION'
    TRADE_DATE = 'TRADE_DATE'
    SETTLEMENT_DATE = 'SETTLEMENT_DATE'

    # Enrichments
    CPTY_ACCOUNT = 'CPTY_ACCOUNT'
    OUR_ACCOUNT = 'OUR_ACCOUNT'
    CPTY_AGENT = 'CPTY_AGENT'
    OUR_AGENT = 'OUR_AGENT'
    CPTY_CONTACT = 'CPTY_CONTACT'
    OUR_CONTACT = 'OUR_CONTACT'
    CPTY_ADDRESS = 'CPTY_ADDRESS'
    OUR_ADDRESS = 'OUR_ADDRESS'


class DateFormats(enum.Enum):
    MMDDYYYY = 'MM-DD-YYYY'
    DDMMYYYY = 'DD/MM-YYYY'
    YYMMDD   = 'YY-MM-DD'
