import unittest


class NotMatchError(RuntimeError):
    pass


class Calculator(object):
    opPool = set("+-*/")
    digitPool = set("0123456789")

    def __init__(self):
        self.stack = list()

    def compile(self, strs):
        res = []

        for s in self.parsedExpression(strs):
            if isinstance(s, (int, float)):
                res.append(s)
            else:
                if s == "(":
                    self.stack.append(s)
                elif s == ")":
                    while True:
                        if len(self.stack) == 0:
                            raise NotMatchError("Brackets not match")
                        tmps = self.stack.pop()
                        if tmps == "(":
                            break
                        else:
                            res.append(tmps)
                else:
                    while len(self.stack) > 0:
                        if self.prioprity(self.stack[-1]) >= self.prioprity(s):
                            res.append(self.stack.pop())
                        else:
                            break
                    self.stack.append(s)
        while len(self.stack) > 0:
            tmp = self.stack.pop()
            if tmp not in self.opPool:
                raise NotMatchError("Brackets not match", tmp)
            res.append(tmp)

        self.res = res
        return self.res

    def run(self):
        if "res" in self.__dir__() and len(self.res) > 0:
            res = self.res
            stack = []
            for e in res:
                if isinstance(e, (int, float)):
                    stack.append(e)
                else:
                    n1 = stack.pop()
                    n2 = stack.pop()
                    r = self._cal(e, n2, n1)
                    stack.append(r)
            return stack.pop()

            # i = 0
            # while len(res) > 1:
            #     if res[i+2] in self.opPool:
            #         a, b, op = res.pop(i), res.pop(i), res.pop(i)
            #         tmpRes = self._cal(op, a, b)
            #         res.insert(i, tmpRes)
            #         i = 0
            #     else:
            #         i += 1

            # return float(res[0])

    def calculate(self, strs):
        self.compile(strs)

        return self.run()

    def _cal(self, op, a, b):
        if op == '+':
            r = a + b
        elif op == '-':
            r = a - b
        elif op == '*':
            r = a * b
        elif op == '/':
            r = a / b
        else:
            raise RuntimeError("Unsupported operator")
        return r

    def parsedExpression(self, strs):
        output = []

        i = 0
        while i < len(strs):
            if strs[i] == " ":
                i += 1
                continue
            elif strs[i] not in self.digitPool:
                output.append(strs[i])
                i += 1
            else:
                tmps = ""
                while (i < len(strs)) and (strs[i] in self.digitPool):
                    tmps += strs[i]
                    i += 1
                output.append(int(tmps))
        return output

    @staticmethod
    def prioprity(op):
        priority = 0
        if op in {"+", "-"}:
            priority = 1
        elif op in {"*", "/"}:
            priority = 2

        return priority


def main():
    expression = "2 * 3 / (25 - 1)+3 * ((4 - 1)"
    cal = Calculator()
    r = cal.calculate(expression)
    print(expression, " -> ", r)
    print(eval(expression) == r)


class test(unittest.TestCase):

    def testEqual(self):
        expression = "2 * 3 / (25 - 1)+3 * (4 - 1)"
        cal = Calculator()
        r = cal.calculate(expression)
        self.assertAlmostEqual(r, 9.25)

    def testNotMatchError(self):
        expression = "2 * 3 / (25 - 1)+3 * ((4 - 1)"
        cal = Calculator()

        with self.assertRaises(NotMatchError):
            _ = cal.compile(expression)


if __name__ == '__main__':
    unittest.main()
