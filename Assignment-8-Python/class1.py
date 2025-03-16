class RomanConverter:
    def int_to_roman(self, num):
        val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        syb = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        roman = ''
        for i in range(len(val)):
            while num >= val[i]:
                num -= val[i]
                roman += syb[i]
        return roman

    def roman_to_int(self, roman):
        rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        num, prev = 0, 0
        for ch in reversed(roman):
            curr = rom_val[ch]
            num = num - curr if curr < prev else num + curr
            prev = curr
        return num

