import os

from gen.error_gen import generate_errors
from gen.full_trades import convert_raw_trades_to_full_trades
from gen.raw_trades import RawTradeGenerator

from gen.utils import delete_files_in_directory, get_argument_parser
from contrastive_matching.utils.utils import get_args

if __name__ == '__main__':

    test_args = get_args()

    # Get and parser arguments
    parser = get_argument_parser()
    args = parser.parse_args()

    print("Arguments parsed. ")

    # Generate Raw Trades
    raw_trades = RawTradeGenerator(args).get_raw_trade_list_random()

    print("Raw trades generated. ")

    # Enrich Raw Trades to Full Trades
    full_trades = convert_raw_trades_to_full_trades(raw_trades)

    # Introduce Errors
    full_trades = generate_errors(full_trades, args)

    print("Full trades created from raw trades. ")

    # Create confirmations
    for full_trade in full_trades:
        full_trade.create_confirmation_from_template()

    print("Confirmations created from full trades and templates ")

    # Change to data directory
    data_path = args.data_path
    delete_files_in_directory(data_path)
    os.chdir(data_path)

    # Save trades and confirmations
    for i, full_trade in enumerate(full_trades):
        full_trade.save_trade_and_confirmation(str(i))

    print(f"{i + 1} new generated trade/confirmation pairs saved in {data_path}. ")

    os.chdir('..')


