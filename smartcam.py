import click
from PIL import Image
import yaml
import cv2

@click.group()
@click.option('-c', '--config', type=click.File(), help='Path to config file.')
@click.pass_context
def main(ctx, config):
    if config:
        try:
            ctx.obj = dict(
                c = yaml.load(config, Loader=yaml.Loader)
            )
        except ImportError:
            click.echo('Error loading config')

@main.command()
@click.pass_context
def detect(ctx):
    config = ctx.obj['c']
    vcap = cv2.VideoCapture(config['url'], cv2.CAP_FFMPEG)

    while (1):
        ret, frame = vcap.read()
        if ret == False:
            print('frame is empty')
            break
        else:
            cv2.imshow('VIDEO', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()