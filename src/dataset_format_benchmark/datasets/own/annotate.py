import argparse
from pathlib import Path
import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader


def get_parsed_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data-root', type=str)

    args, args_other = parser.parse_known_args()

    return args


def main():
    args = get_parsed_args()
    data_root_path = Path(args.data_root)

    loader = QUiLoader()

    app = QtWidgets.QApplication(sys.argv)
    window = loader.load(Path(__file__).parent / 'annotation_window.ui', None)
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
