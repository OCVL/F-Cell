import parse

from ocvl.function.utility.json_format_constants import DataFormat


class FormatParser():
    def __init__(self, format_string):
        self.formatlocs = dict()
        self.file_parse = parse.compile(format_string)


    def parse_file(self, file_string):

        filename_metadata = dict()
        print(self.file_parse)
        ext_str = self.file_parse.parse(file_string)
        # Need to parse using each of our types to determine the format it matches

        for formatstr in DataFormat:
            if formatstr.value in ext_str.named:
                filename_metadata[formatstr] = ext_str[formatstr.value]
            else:
                print("Didn't find "+ formatstr.value +" in the parsed file.")

        return filename_metadata