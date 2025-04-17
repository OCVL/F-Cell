import parse
from parse_type import TypeBuilder

from ocvl.function.utility.json_format_constants import DataTags, DataFormatType


class FormatParser():
    def __init__(self, video_format=None, mask_format=None, image_format=None, metadata_format=None, queryloc_format=None):
        self.formatlocs = dict()

        # An optional parser for strings.
        self.optional_parse = TypeBuilder.with_optional(lambda opt_str: str(opt_str))

        if video_format is not None:
            self.vid_parser = parse.compile(video_format, {"s?":self.optional_parse})
        else:
            self.vid_parser = None

        if mask_format is not None:
            self.mask_parser = parse.compile(mask_format, {"s?":self.optional_parse})
        else:
            self.mask_parser = None

        if image_format is not None:
            self.im_parser = parse.compile(image_format, {"s?":self.optional_parse})
        else:
            self.im_parser = None

        if queryloc_format is not None:
            self.queryloc_parser = parse.compile(queryloc_format, {"s?":self.optional_parse})
        else:
            self.queryloc_parser = None

        if metadata_format is not None:
            self.metadata_parser = parse.compile(metadata_format, {"s?":self.optional_parse})
        else:
            self.metadata_parser = None

    def parse_file(self, file_string):

        filename_metadata = dict()

        parsed_str = self.vid_parser.parse(file_string)
        parser_used = DataFormatType.VIDEO
        if parsed_str is None and self.mask_parser is not None:
            parsed_str = self.mask_parser.parse(file_string)
            parser_used = DataFormatType.MASK
        if parsed_str is None and self.im_parser is not None:
            parsed_str = self.im_parser.parse(file_string)
            parser_used = DataFormatType.IMAGE
        if parsed_str is None and self.queryloc_parser is not None:
            parsed_str = self.queryloc_parser.parse(file_string)
            parser_used = DataFormatType.QUERYLOC
        if parsed_str is None and self.metadata_parser is not None:
            parsed_str = self.metadata_parser.parse(file_string)
            parser_used = DataFormatType.METADATA
        if parsed_str is None:
            return None, filename_metadata


        for formatstr in DataTags:
            if formatstr in parsed_str.named:
                if parsed_str[formatstr] is not None:
                    filename_metadata[formatstr.value] = parsed_str[formatstr]
                else:
                    filename_metadata[formatstr.value] = ""


        return parser_used, filename_metadata