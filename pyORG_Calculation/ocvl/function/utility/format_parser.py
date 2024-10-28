import parse

from ocvl.function.utility.json_format_constants import DataFormat, FormatTypes


class FormatParser():
    def __init__(self, video_format, mask_format=None, image_format=None, metadata_format=None):
        self.formatlocs = dict()
        self.vid_parser = parse.compile(video_format)
        if mask_format:
            self.mask_parser = parse.compile(mask_format)
        if image_format:
            self.im_parser = parse.compile(image_format)
        if metadata_format:
            self.metadata_parser = parse.compile(metadata_format)


    def parse_file(self, file_string):

        filename_metadata = dict()

        parsed_str = self.vid_parser.parse(file_string)
        parser_used = FormatTypes.VIDEO
        if not parsed_str:
            parsed_str = self.mask_parser.parse(file_string)
            parser_used = FormatTypes.MASK
        if not parsed_str:
            parsed_str = self.im_parser.parse(file_string)
            parser_used = FormatTypes.IMAGE
        if not parsed_str:
            parsed_str = self.metadata_parser.parse(file_string)
            parser_used = FormatTypes.METADATA
        if not parsed_str:
            return None, filename_metadata


        for formatstr in DataFormat:
            if formatstr in parsed_str.named:
                filename_metadata[formatstr.value] = parsed_str[formatstr]

        return parser_used, filename_metadata