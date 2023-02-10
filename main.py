from importlib import import_module
from params import parser
import models.model as model

args = parser.parse_args()
module = import_module(args.mode)
_model = getattr(model, dir(model)[0])(args)

if __name__ == '__main__':
    getattr(module, args.mode)(_model, args)
