from ocvl.function.utility.json_format_constants import DataFormat


class FormatParser():
    def __init__(self, format_string, separator):
        self.formatlocs = dict()
        self.max_loc = len(format_string)
        self.separator = separator

        loc_ind = 0
        for chunk in format_string.split(separator):
            print(chunk)
            for formatstr in DataFormat:
                if formatstr.value in chunk and loc_ind not in self.formatlocs:
                    self.formatlocs[loc_ind] = (formatstr, )
                elif formatstr.value in chunk and loc_ind in self.formatlocs:
                    # Concat other formats to the tuple associated with this chunk- this is implicitly in the order they appear.
                    self.formatlocs[loc_ind] = self.formatlocs[loc_ind] + (formatstr,)

            loc_ind += 1
        print(self.formatlocs)


    def parse_file(self, file_string):

        filename_metadata = dict()

        loc_ind = 0
        for chunk in file_string.split(self.separator):
            self.formatlocs[loc_ind]
            loc_ind += 1

        for key, value in self.formatlocs.items():
            filename_metadata[key] = file_string[value[0]:value[0]+value[1]]

        return filename_metadata