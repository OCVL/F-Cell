import parse

from ocvl.function.utility.json_format_constants import DataTags, DataType


class FormatParser():
    def __init__(self, video_format, mask_format=None, image_format=None, metadata_format=None, queryloc_format=None):
        self.formatlocs = dict()
        self.vid_parser = parse.compile(video_format)

        if mask_format is not None:
            self.mask_parser = parse.compile(mask_format)
        else:
            self.mask_parser = None

        if image_format is not None:
            self.im_parser = parse.compile(image_format)
        else:
            self.im_parser = None

        if queryloc_format is not None:
            self.queryloc_parser = parse.compile(queryloc_format)
        else:
            self.queryloc_parser = None

        if metadata_format is not None:
            self.metadata_parser = parse.compile(metadata_format)
        else:
            self.metadata_parser = None


    def parse_file(self, file_string):

        filename_metadata = dict()

        parsed_str = self.vid_parser.parse(file_string)
        parser_used = DataType.VIDEO
        if parsed_str is None and self.mask_parser is not None:
            parsed_str = self.mask_parser.parse(file_string)
            parser_used = DataType.MASK
        if parsed_str is None and self.im_parser is not None:
            parsed_str = self.im_parser.parse(file_string)
            parser_used = DataType.IMAGE
        if parsed_str is None and self.queryloc_parser is not None:
            parsed_str = self.queryloc_parser.parse(file_string)
            parser_used = DataType.QUERYLOC
        if parsed_str is None and self.metadata_parser is not None:
            parsed_str = self.metadata_parser.parse(file_string)
            parser_used = DataType.METADATA
        if parsed_str is None:
            return None, filename_metadata


        for formatstr in DataTags:
            if formatstr in parsed_str.named:
                filename_metadata[formatstr.value] = parsed_str[formatstr]


        return parser_used, filename_metadata