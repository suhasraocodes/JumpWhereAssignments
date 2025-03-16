class StringHandler:
    def get_string(self):
        self.s = input('Enter string: ')
    def print_string(self):
        print(self.s[::-1])
