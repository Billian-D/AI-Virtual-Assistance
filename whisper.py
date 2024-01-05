import asyncio
import ctypes
import logging
import pathlib
import platform
from signal import SIGINT, SIGTERM
import json
import requests
import pygame

from gtts import gTTS
from io import BytesIO

import numpy as np
from livekit import api, rtc

platform = platform.system().lower()
if platform == "windows":
    lib_file = "whisper.dll"
elif platform == "darwin":
    lib_file = "libwhisper.dylib"
else:
    lib_file = "libwhisper.so"

whisper_dir = pathlib.Path(__file__).parent.absolute() / "whisper.cpp"
libname = str(whisper_dir / lib_file)
fname_model = str(whisper_dir / "models/ggml-tiny.en.bin")

# declare the Whisper C API  (Only what we need, keep things simple)
# also see this issue: https://github.com/ggerganov/whisper.cpp/issues/9
# structure must match https://github.com/ggerganov/whisper.cpp/blob/master/whisper.h


class WhisperSamplingStrategy(ctypes.c_int):
    WHISPER_SAMPLING_GREEDY = 0
    WHISPER_SAMPLING_BEAM_SEARCH = 1


class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
        ("speed_up", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        ("tdrz_enable", ctypes.c_bool),
        ("initial_prompt", ctypes.c_char_p),
        ("prompt_tokens", ctypes.c_void_p),
        ("prompt_n_tokens", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("detect_language", ctypes.c_bool),
        ("suppress_blank", ctypes.c_bool),
        ("suppress_non_speech_tokens", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        ("greedy", ctypes.c_int),
        ("beam_size", ctypes.c_int),
        ("patience", ctypes.c_float),
        ("new_segment_callback", ctypes.c_void_p),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", ctypes.c_void_p),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", ctypes.c_void_p),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
    ]


WHISPER_SAMPLE_RATE = 16000
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
SAMPLES_30_SECS = WHISPER_SAMPLE_RATE * 30
SAMPLES_KEEP = WHISPER_SAMPLE_RATE * 1  # data to keep from the old inference
SAMPLES_STEP = WHISPER_SAMPLE_RATE * 3  # 3 seconds of new data

whisper = ctypes.CDLL(libname)
whisper.whisper_init_from_file.argtypes = [ctypes.c_char_p]
whisper.whisper_init_from_file.restype = ctypes.c_void_p
whisper.whisper_full_default_params.restype = WhisperFullParams
whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p
ctx = whisper.whisper_init_from_file(fname_model.encode("utf-8"))


async def main(room: rtc.Room):
    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logging.info("Participant %s has disconnected", participant.identity)
        asyncio.ensure_future(cleanup())

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        # Only subscribe to the audio tracks coming from the microphone
        if (
            publication.kind == rtc.TrackKind.KIND_AUDIO
            and publication.source == rtc.TrackSource.SOURCE_MICROPHONE
        ):
            logging.info(
                "track published: %s from participant %s (%s), subscribing...",
                publication.sid,
                participant.sid,
                participant.identity,
            )

            publication.set_subscribed(True)

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logging.info("starting listening to: %s", participant.identity)
        audio_stream = rtc.AudioStream(track)
        asyncio.create_task(whisper_task(audio_stream, participant))

    url = "ws://localhost:7880"
    token = (
        api.AccessToken(api_key="devkey", api_secret="secret")
        .with_identity("KITT")
        .with_name("KITT")
        .with_grants(api.VideoGrants(room_join=True, room="s4g2-nbph"))
        .to_jwt()
    )
    # manually manage subscriptions
    await room.connect(url, token, rtc.RoomOptions(auto_subscribe=False))
    logging.info("connected to room %s", room.name)

    # check if there are already published audio tracks
    for participant in room.participants.values():
        for track in participant.tracks.values():
            if (
                track.kind == rtc.TrackKind.KIND_AUDIO
                and track.source == rtc.TrackSource.SOURCE_MICROPHONE
            ):
                track.set_subscribed(True)


async def whisper_task(
    stream: rtc.AudioStream,
    participant: rtc.RemoteParticipant,
):
    data_30_secs = np.zeros(SAMPLES_30_SECS, dtype=np.float32)
    written_samples = 0  # nb. of samples written to data_30_secs for the cur. inference

    async for frame in stream:
        # whisper requires 16kHz mono, so resample the data
        # also convert the samples from int16 to float32

        frame = frame.remix_and_resample(WHISPER_SAMPLE_RATE, 1)

        data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

        # write the data inside data_30_secs at written_samples
        data_start = SAMPLES_KEEP + written_samples
        data_30_secs[data_start : data_start + len(data)] = data
        written_samples += len(data)

        if written_samples >= SAMPLES_STEP:
            params = whisper.whisper_full_default_params(
                WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY
            )
            params.print_realtime = False
            params.print_progress = False

            ctx_ptr = ctypes.c_void_p(ctx)
            data_ptr = data_30_secs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            res = whisper.whisper_full(
                ctx_ptr, params, data_ptr, written_samples + SAMPLES_KEEP
            )

            if res != 0:
                logging.error("error while running inference: %s", res)
                return

            n_segments = whisper.whisper_full_n_segments(ctx_ptr)
            for i in range(n_segments):
                t0 = whisper.whisper_full_get_segment_t0(ctx_ptr, i)
                t1 = whisper.whisper_full_get_segment_t1(ctx_ptr, i)
                txt = whisper.whisper_full_get_segment_text(ctx_ptr, i)

                # speech to text convert
                participant_transcribed_text = txt.decode("utf-8")
                logging.info(
                    f"{t0/1000.0:.3f} - {t1/1000.0:.3f} : {txt.decode('utf-8')}"
                )
                data_payload = json.dumps(
                    {
                        "data": {
                            "sid": participant.sid,
                            "name": participant.name,
                            "text": participant_transcribed_text,
                            "state": 3,
                        },
                        "type": 0,
                    }
                )
                data = data_payload.encode("utf-8")

                # publish participant data into subtitle
                await room.local_participant.publish_data(data, 3)

                invalid_text = [
                    " [BLANK_AUDIO]",
                    " [Music]",
                    " [MUSIC PLAYING]",
                    " [MUSIC]",
                    " (dramatic music)",
                    " [unintelligible]",
                    " [inaudible]",
                    " (buzzing)",
                    " (zipper buzzing)",
                    " *sigh*",
                    " [SOUND]",
                    " [coughing]",
                    " [Chuckles]",
                    " (static)",
                    " (playful music)",
                    " (bells chiming)",
                    " (laughing)",
                    " (indistinct chatter)",
                    " (cheerful music)"
                    " [INAUDIBLE]"
                ]

                if participant_transcribed_text not in invalid_text:
                    # Prepare privategpt API call
                    api_url = "http://35.244.13.63:8001/v1/completions"
                    api_data = {
                        "include_sources": False,
                        "prompt": participant_transcribed_text,
                        "stream": False,
                        "use_context": False,
                    }

                    # Make API call
                    try:
                        response = requests.post(api_url, json=api_data)
                        response.raise_for_status()
                        logging.info(
                            "Privategpt API call successful. Response: %s",
                            response.json()["choices"][0]["message"]["content"],
                        )

                        publication_transcribed_text = response.json()["choices"][0][
                            "message"
                        ]["content"]

                        source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
                        track = rtc.LocalAudioTrack.create_audio_track(
                            "sinewave", source
                        )
                        options = rtc.TrackPublishOptions()
                        options.source = rtc.TrackSource.SOURCE_MICROPHONE
                        publication = await room.local_participant.publish_track(
                            track, options
                        )
                        logging.info("published track %s", publication.sid)

                        await text_to_speech(publication_transcribed_text)
                    except requests.exceptions.RequestException as e:
                        logging.error("Error making privategpt API call: %s", e)

            # write old data to the beginning of the buffer (SAMPLES_KEEP)
            data_30_secs[:SAMPLES_KEEP] = data_30_secs[
                data_start
                + written_samples
                - SAMPLES_KEEP : data_start
                + written_samples
            ]
            written_samples = 0

async def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language)
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)

    pygame.mixer.init()
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("whisper.log"), logging.StreamHandler()],
    )

    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup():
        await room.disconnect()
        loop.stop()

    asyncio.ensure_future(main(room))
    for signal in [SIGINT, SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
