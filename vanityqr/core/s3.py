from PIL import Image

from botocore.exceptions import NoCredentialsError, ClientError

from core import util
import re, boto3, sys, os, io

sys.path.append(os.path.abspath('../'))
import settings


def clean_s3_file( s3_file ):
    return re.sub('^/', '', util.xstr(s3_file))


def url( url ):
    if url is None:
        return ""

    url = util.xstr(url)
    if url == "":
        return ""

    return "https://%s.%s.%s/%s" % (settings.S3_ACCESS['BUCKET'],
                                    settings.S3_ACCESS['REGION'],
                                    settings.S3_ACCESS['HOST'],
                                    clean_s3_file(url))


def put_data(data, s3_file, bucket=None):
    # Default bucket
    if bucket is None:
        bucket = settings.S3_ACCESS['BUCKET']

    # Dev code
    if 'MODE' in settings.S3_ACCESS and settings.S3_ACCESS['MODE'] == 'DEV':
        endpoint = "https://%s.%s" % (settings.S3_ACCESS['REGION'], settings.S3_ACCESS['HOST'])

        # Initialize a session using DigitalOcean Spaces.
        session = boto3.session.Session()
        client = session.client('s3',
                                region_name=settings.S3_ACCESS['REGION'],
                                endpoint_url=endpoint,
                                aws_access_key_id=settings.S3_ACCESS['ACCESS_KEY'],
                                aws_secret_access_key=settings.S3_ACCESS['SECRET_KEY'])

    # Production
    else:
        client = boto3.client("s3")

    try:
        args = { "Bucket": bucket, "Key": clean_s3_file(s3_file), "Body": data }
        if 'EXTRA_ARGS' in settings.S3_ACCESS and util.xstr(settings.S3_ACCESS['EXTRA_ARGS']) != "":
            args["ACL"] = settings.S3_ACCESS['EXTRA_ARGS']
        ret = client.put_object( **args )
        return ret['ResponseMetadata']['HTTPStatusCode'] == 200

    except FileNotFoundError:
        print("The file was not found")
    except NoCredentialsError:
        print("Credentials not available")
    except ClientError:
        print("AccessDenied")

    return False


def put_image( image: Image, s3_file, bucket=None ):
    # Create a BytesIO object
    byte_stream = io.BytesIO()

    # Save the image to the byte stream
    image.save(byte_stream, format='PNG')

    # Retrieve the bytes from the stream
    image_bytes = byte_stream.getvalue()
    result = put_data( image_bytes, s3_file, bucket )

    # Close the byte stream
    byte_stream.close()

    return result