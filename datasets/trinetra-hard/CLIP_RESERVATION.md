# CLIP_RESERVATION — indian_road

Published: 2026-09-05. **FOR MANAS.**

**Rule: split by `clip_id`, never by frame** (CLAUDE.md SS5). Same-clip frames
are near-duplicates; perceptual hashing will not catch them, and a frame-level
split silently leaks eval data into training.

**Scope:** only 5 of 646 upstream WebDataset shards are downloaded locally
(`data/train-00000-of-00646.tar` .. `train-00004-of-00646.tar`). This reservation
covers the 62 clips / 5,000 frames physically present in this checkout, not the
full 8,441-clip corpus. If more shards are downloaded later, treat every new
clip_id as **reserved by default** until this file is explicitly updated —
do not assume TRAIN_SAFE for anything not listed below.

**Sizing:** TRINETRA-HARD targets 300 candidate observations total (across all
9 raw datasets, not indian_road alone). Worst case, indian_road supplies all 300.
Reserved here: **31 clips / 2,460 frames** — roughly 4x a 2x-buffered 600-frame
requirement (300 obs x 2 for safety margin). Over-reserved deliberately per
instruction: under-reserving is unrecoverable (a trained-on clip can never be
used for eval again), over-reserving only costs Manas some training frames.

**Totals:** 62 distinct clip_id values, 5000 frames.
RESERVED: 31 clips / 2460 frames.
TRAIN_SAFE: 31 clips / 2540 frames.
Check: 31 + 31 = 62 (== 62); 2460 + 2540 = 5000 (== 5000).

## RESERVED — held for evaluation, never trained on

| clip_id | frames |
|---|---|
| 000db725-8180-4770-8b6a-4eb74aeda9f9 | 60 |
| 000ebc57-d7e6-4e62-97a6-2cbc399724a1 | 60 |
| 0016b97d-ddc3-4d5f-8e7f-414fd7240399 | 60 |
| 0019389a-f7aa-4696-bc89-5f8762fc4c77 | 60 |
| 001c0d67-590e-479b-86b9-c521fe884139 | 60 |
| 001fd88e-ce16-4f22-b005-34f7791fb996 | 60 |
| 00263864-d94b-44c5-8eb9-4e51093e59f4 | 60 |
| 0031e84e-43d4-4457-b85b-d90163c7267e | 60 |
| 003666c8-18f8-46c3-96a7-a536ab0d25c4 | 60 |
| 003bf0b5-6eb4-4359-bffa-3debea8e6e5e | 60 |
| 0041845c-79d8-49b6-a7df-1f2cbbd79f9e | 60 |
| 0049d23b-5aaf-4bc4-9f9c-a9c2c03dbc9e | 180 |
| 004c8b8d-11af-44a9-9e61-a4e34e643006 | 180 |
| 004d1e7c-4dbe-405d-8782-598ca0da1bea | 60 |
| 00539b87-f629-4661-9793-01ab7905383e | 60 |
| 0054a12a-f4f0-41de-aa26-566097787891 | 60 |
| 0076d1e2-ef37-4445-9fa9-7d3bdf25bd8f | 60 |
| 0078bb37-7ca2-4bdd-8563-626117130fb7 | 180 |
| 00797c49-b024-47d9-85e7-e7391d1498f7 | 60 |
| 007cedf1-aefd-4129-b852-7695829a4cec | 60 |
| 007e2cdb-abe0-4fe7-89ea-79569159ad14 | 180 |
| 008062ff-e7f8-458b-80bd-1f909551011c | 60 |
| 0088ab0e-9098-4ec5-8f36-a47aef7ae54e | 60 |
| 00c87e36-15a5-4ca8-9f01-5ebbaadf11b4 | 60 |
| 00ca74e0-be2b-4b3e-ab4b-46da5a9f4279 | 60 |
| 00cca8c2-9bcf-4c11-a245-fdc3d5ee2769 | 60 |
| 00d5614c-7cd1-4263-8db4-bf89c75c34bf | 60 |
| 00f7ff41-16c5-45fe-a970-f05542e39b56 | 180 |
| 01074748-cb68-43b9-a3f1-86ce2bdc0bed | 60 |
| 01143ada-8e82-4812-bfc6-63842c9f4dc6 | 60 |
| 01178f2d-4186-4c7f-a284-de99f4ebdbe4 | 60 |

## TRAIN_SAFE — every remaining clip_id, safe for Manas to train on

| clip_id | frames |
|---|---|
| 0123b6b9-7e2b-4eaa-8e99-d484a56b22bf | 60 |
| 0126ba36-e59a-4dc9-b8b3-b2af8dc8c7b4 | 60 |
| 012b6a2b-d657-48fc-95fd-6b1743f12302 | 180 |
| 012fd9eb-f2e8-42c8-818f-2a77081e3f49 | 60 |
| 013aa0bd-3865-40b9-91ca-942396da6e89 | 60 |
| 01471f07-5161-4c16-92e5-83a5abe9f27c | 60 |
| 015fe621-a79b-44fc-b8bc-9df5444601ec | 60 |
| 0168ba4f-0887-4416-9bbc-7e4864407eaa | 60 |
| 016a624e-fcb8-41eb-83d8-e86fd38f495b | 60 |
| 017b58f8-2209-4d52-948e-42229f3727a6 | 180 |
| 0184dbbd-e3b7-4246-b36d-d2baa8a69b8d | 60 |
| 018a1b44-b9ab-4566-908a-acb4454ba889 | 60 |
| 0192f09c-4472-42a7-8589-9b2e6b3ab4de | 60 |
| 0193588d-2bf1-4dbd-a3ef-6b129e961e32 | 180 |
| 01977365-1007-4f84-88fd-eb6ca0326ac4 | 60 |
| 01a6d089-8600-41d3-901f-e85a6e9fc582 | 180 |
| 01ad7e37-95ef-4182-a8c1-c1620723d80a | 60 |
| 01b55cf0-c6eb-4c78-bf53-c425cb275b8e | 60 |
| 01b9aaa0-5c0b-41ec-9691-224bcb788f44 | 60 |
| 01ba8c48-8396-46e8-a257-e791320d7476 | 60 |
| 01bda163-9e82-4ca2-ab38-049876d9bca8 | 180 |
| 01d46ccf-1cc7-4dc9-9cd7-f8ebc85a8a0c | 60 |
| 01dcad3e-5564-4b0f-a555-8bdeba61864d | 180 |
| 01e8aa66-efbd-41be-a2f3-354d01103755 | 60 |
| 01f648eb-4fba-46a2-b471-5f56c95c434d | 60 |
| 020e23ed-7195-47c3-85e7-123772977eb0 | 60 |
| 0218bfe0-c2fb-41fc-8b98-833683334356 | 60 |
| 0219e7d0-cedd-4f01-88fb-d6f7369653fa | 60 |
| 021a3a6d-445b-475f-8ad7-f2211ff38e55 | 60 |
| 021a839a-4013-42ab-889f-130bfbb1a318 | 60 |
| 021bbeda-bd80-45f7-83ff-aa514beb2be5 | 20 |
