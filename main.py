import os
import whisperx
import magic
import argparse


def main():

    parser = argparse.ArgumentParser(description='Transcribe audio files to text.')
    parser.add_argument('-o', '--output', help='output to text file (default: output to stdout)')
    parser.add_argument('files', nargs='+', help='files to transcribe')
    args = parser.parse_args()

    files_to_transcribe = []
    for file in args.files:
        if os.path.isfile(file):
            if 'audio' in magic.from_file(file, mime=True):
                files_to_transcribe.append(file)

    compute_type = 'float32'
    device = 'cpu'

    f = None
    if args.output:
        f = open(args.output, 'w', encoding='utf-8')

    model = whisperx.load_model('large-v2', device=device, compute_type=compute_type)
    for file_to_transcribe in files_to_transcribe:
        if f:
            f.write(f'file transcribed: {file_to_transcribe}\n')
        audio = whisperx.load_audio(file_to_transcribe)
        result = model.transcribe(audio)
        text = ''
        for segment in result['segments']:
            text = text + segment['text']
        if f:
            f.write(text.strip() + '\n\n')
        else:
            print(text.strip())
    if f:
        f.close()


if __name__ == '__main__':
    main()

