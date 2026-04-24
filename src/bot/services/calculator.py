import math
import re


class CalculatorError(ValueError):
    pass


def _subfactorial(value: int) -> int:
    if value < 0:
        raise CalculatorError("Subfactorial is defined only for non-negative integers.")
    if value == 0:
        return 1
    if value == 1:
        return 0
    prev2, prev1 = 1, 0
    for current in range(2, value + 1):
        prev2, prev1 = prev1, (current - 1) * (prev1 + prev2)
    return prev1


class _Parser:
    def __init__(self, expression: str):
        self.expression = expression
        self.length = len(expression)
        self.position = 0

    def parse(self) -> float:
        value = self._parse_expression()
        self._skip_spaces()
        if self.position != self.length:
            raise CalculatorError(f"Unexpected token at position {self.position + 1}.")
        return value

    def _skip_spaces(self) -> None:
        while self.position < self.length and self.expression[self.position].isspace():
            self.position += 1

    def _peek(self, token: str) -> bool:
        self._skip_spaces()
        return self.expression.startswith(token, self.position)

    def _consume(self, token: str) -> bool:
        if self._peek(token):
            self.position += len(token)
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._consume(token):
            raise CalculatorError(f"Expected '{token}' at position {self.position + 1}.")

    def _parse_expression(self) -> float:
        value = self._parse_term()
        while True:
            if self._consume("+"):
                value += self._parse_term()
            elif self._consume("-"):
                value -= self._parse_term()
            else:
                return value

    def _parse_term(self) -> float:
        value = self._parse_power()
        while True:
            if self._consume("//"):
                right = self._parse_power()
                if right == 0:
                    raise CalculatorError("Division by zero.")
                value = value // right
            elif self._consume("*"):
                value *= self._parse_power()
            elif self._consume("/"):
                right = self._parse_power()
                if right == 0:
                    raise CalculatorError("Division by zero.")
                value /= right
            elif self._consume("%"):
                right = self._parse_power()
                if right == 0:
                    raise CalculatorError("Division by zero.")
                value %= right
            else:
                return value

    def _parse_power(self) -> float:
        value = self._parse_unary()
        if self._consume("**"):
            exponent = self._parse_power()
            value = value**exponent
        return value

    def _parse_unary(self) -> float:
        if self._consume("+"):
            return self._parse_unary()
        if self._consume("-"):
            return -self._parse_unary()
        if self._consume("!"):
            value = self._parse_unary()
            integer = _coerce_non_negative_int(value, "Subfactorial")
            return float(_subfactorial(integer))
        return self._parse_postfix()

    def _parse_postfix(self) -> float:
        value = self._parse_primary()
        while self._consume("!"):
            integer = _coerce_non_negative_int(value, "Factorial")
            value = float(math.factorial(integer))
        return value

    def _parse_primary(self) -> float:
        self._skip_spaces()
        if self._consume("("):
            value = self._parse_expression()
            self._expect(")")
            return value

        if self.position >= self.length:
            raise CalculatorError("Unexpected end of expression.")

        if self.expression[self.position].isdigit() or self.expression[self.position] == ".":
            return self._parse_number()

        if self.expression[self.position].isalpha() or self.expression[self.position] == "_":
            name = self._parse_identifier()
            if self._consume("("):
                args = self._parse_arguments()
                return _call_function(name, args)
            return _resolve_constant(name)

        raise CalculatorError(f"Unexpected token at position {self.position + 1}.")

    def _parse_number(self) -> float:
        start = self.position
        dot_seen = False
        while self.position < self.length:
            char = self.expression[self.position]
            if char.isdigit():
                self.position += 1
                continue
            if char == "." and not dot_seen:
                dot_seen = True
                self.position += 1
                continue
            break
        try:
            return float(self.expression[start:self.position])
        except ValueError as error:
            raise CalculatorError("Invalid number.") from error

    def _parse_identifier(self) -> str:
        start = self.position
        while self.position < self.length and (
            self.expression[self.position].isalnum() or self.expression[self.position] == "_"
        ):
            self.position += 1
        return self.expression[start:self.position].lower()

    def _parse_arguments(self) -> list[float]:
        args: list[float] = []
        if self._consume(")"):
            return args
        while True:
            args.append(self._parse_expression())
            if self._consume(")"):
                return args
            self._expect(",")


def _coerce_non_negative_int(value: float, label: str) -> int:
    if isinstance(value, float) and not value.is_integer():
        raise CalculatorError(f"{label} is defined only for integers.")
    integer = int(value)
    if integer < 0:
        raise CalculatorError(f"{label} is defined only for non-negative integers.")
    return integer


def _resolve_constant(name: str) -> float:
    constants = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }
    if name not in constants:
        raise CalculatorError(f"Unknown constant: {name}.")
    return constants[name]


def _call_function(name: str, args: list[float]) -> float:
    if name == "sqrt":
        _expect_arity(name, args, 1)
        if args[0] < 0:
            raise CalculatorError("sqrt() is defined only for non-negative values.")
        return math.sqrt(args[0])
    if name == "abs":
        _expect_arity(name, args, 1)
        return abs(args[0])
    if name == "sin":
        _expect_arity(name, args, 1)
        return math.sin(args[0])
    if name == "cos":
        _expect_arity(name, args, 1)
        return math.cos(args[0])
    if name == "tan":
        _expect_arity(name, args, 1)
        return math.tan(args[0])
    if name == "ln":
        _expect_arity(name, args, 1)
        return math.log(args[0])
    if name == "log":
        if len(args) == 1:
            return math.log10(args[0])
        if len(args) == 2:
            return math.log(args[0], args[1])
        raise CalculatorError("log() expects one or two arguments.")
    if name == "exp":
        _expect_arity(name, args, 1)
        return math.exp(args[0])
    if name == "floor":
        _expect_arity(name, args, 1)
        return float(math.floor(args[0]))
    if name == "ceil":
        _expect_arity(name, args, 1)
        return float(math.ceil(args[0]))
    if name == "round":
        if len(args) == 1:
            return float(round(args[0]))
        if len(args) == 2:
            return float(round(args[0], int(args[1])))
        raise CalculatorError("round() expects one or two arguments.")
    if name == "factorial":
        _expect_arity(name, args, 1)
        return float(math.factorial(_coerce_non_negative_int(args[0], "Factorial")))
    if name == "subfactorial":
        _expect_arity(name, args, 1)
        return float(_subfactorial(_coerce_non_negative_int(args[0], "Subfactorial")))
    raise CalculatorError(f"Unknown function: {name}.")


def _expect_arity(name: str, args: list[float], expected: int) -> None:
    if len(args) != expected:
        raise CalculatorError(f"{name}() expects {expected} argument(s).")


def calculate_expression(expression: str) -> str:
    normalized = expression.strip()
    if not normalized:
        raise CalculatorError("Expression is empty.")

    if len(normalized) > 300:
        raise CalculatorError("Expression is too long.")

    if re.search(r"[^0-9A-Za-z_+\-*/%().,! \t\r\n]", normalized):
        raise CalculatorError("Expression contains unsupported characters.")

    value = _Parser(normalized).parse()
    if isinstance(value, float) and value.is_integer():
        rendered = str(int(value))
    else:
        rendered = format(value, ".12g")

    return (
        f"expression: {normalized}\n"
        f"result: {rendered}\n"
        "notation: postfix n! means factorial, prefix !n means subfactorial"
    )
