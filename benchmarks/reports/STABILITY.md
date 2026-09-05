# STABILITY — real-footage diagnostic

**This is a stability diagnostic, not an accuracy measurement.** indian_road's `unverified_real` rows have no known true plate string (`eligible: false` always) -- correctness is undefined here. This table only asks whether fusion makes predictions more self-consistent across frames of the same TrackKey. It carries no accuracy claim; see `FUSION_DELTA.md` for that (synthetic_truth only).

Tracks: 580 | Frames: 4812 | Tracks with improved agreement (fusion on vs off): 550/580

Mean agreement -- fusion OFF: 0.455 | fusion ON: 1.000

| TrackKey (camera, session, track) | n_frames | agreement OFF | agreement ON | improved |
|---|---|---|---|---|
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 2 | 8 | 0.375 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 9 | 4 | 0.75 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 15 | 11 | 0.273 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 17 | 11 | 0.364 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 36 | 8 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 38 | 6 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 64 | 7 | 0.286 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 69 | 5 | 0.4 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 80 | 6 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 86 | 6 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 88 | 4 | 0.75 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 94 | 7 | 0.429 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 96 | 9 | 0.444 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 99 | 7 | 0.571 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 114 | 8 | 0.375 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 121 | 3 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 143 | 3 | 0.667 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 154 | 2 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 158 | 7 | 0.286 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 163 | 8 | 0.375 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 168 | 2 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 176 | 13 | 0.385 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 201 | 10 | 0.3 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 248 | 3 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 256 | 5 | 0.4 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 275 | 4 | 0.25 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 280 | 5 | 0.4 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 310 | 6 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 314 | 5 | 0.4 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 315 | 3 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 331 | 6 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 365 | 2 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 377 | 5 | 0.2 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 417 | 4 | 0.25 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 426 | 4 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 441 | 5 | 0.4 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 448 | 2 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 485 | 2 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 499 | 3 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 545 | 3 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 546 | 2 | 1.0 | 1.0 | no |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 547 | 5 | 0.4 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 552 | 3 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 565 | 2 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 595 | 3 | 0.333 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 640 | 2 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 643 | 2 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 707 | 3 | 0.667 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 708 | 2 | 0.5 | 1.0 | yes |
| th_cam_cpplus_dashcam_fleet / 000db725-8180-4770-8b6a-4eb74aeda9f9 / 709 | 2 | 0.5 | 1.0 | yes |

... 530 more tracks omitted from this table, see STABILITY.json
