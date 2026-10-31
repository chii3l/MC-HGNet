"""Exact CRHE cleaning and metadata-only partition rules used in the paper."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
BANDS=('green','red','ir')
SEED=20261020
FOLDS=5
EMBARGO_SECONDS=1800
def load_raw(root):
    RAW=Path(root)
    grouped = defaultdict(lambda: defaultdict(list))
    row_count = 0
    for path in sorted((RAW / "CRHE").glob("ID*/*/*.csv")):
        relative = path.relative_to(RAW).as_posix()
        subject, band = path.parts[-3:-1]
        original_split = path.stem.rsplit("_", 1)[-1]
        frame = pd.read_csv(
            path, dtype={"id": str, "datetime": str}, float_precision="round_trip"
        )
        waveform_columns = [str(i) for i in range(320)]
        assert set(frame.columns) == set(waveform_columns) | {
            "id", "datetime", "gender", "age", "SBP", "DBP", "category"
        }, relative
        for index, row in frame.iterrows():
            timestamp = datetime.strptime(row["datetime"], "%Y-%m-%d %H-%M")
            waveform = row[waveform_columns].to_numpy(dtype="<f8")
            source_id = str(row["id"])
            record = dict(
                subject=subject, band=band, timestamp=timestamp,
                original_split=original_split, source_file=relative,
                csv_row=int(index) + 2,  # Header is row 1.
                sbp=float(row["SBP"]), dbp=float(row["DBP"]),
                age=float(row["age"]), sex=float(row["gender"]),
                identity_hash=hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
                waveform=waveform,
                waveform_sha256=hashlib.sha256(waveform.tobytes()).hexdigest(),
            )
            grouped[(subject, timestamp)][band].append(record)
            row_count += 1
    return grouped, row_count


def record_id(key):
    subject, timestamp = key
    return subject + "_" + timestamp.strftime("%Y%m%dT%H%M")


def clean(grouped):
    """Only exact input validity/identity rules; no learned or target-range filter."""
    wave_locations = defaultdict(set)
    for key, channels in grouped.items():
        for band, rows in channels.items():
            for row in rows:
                wave_locations[(band, row["waveform_sha256"])].add(key)
    reused = {
        key for locations in wave_locations.values() if len(locations) > 1
        for key in locations
    }
    kept = []
    excluded = []
    exact_duplicate_rows_removed = 0
    for key, channels in sorted(grouped.items()):
        subject, timestamp = key
        all_rows = [row for rows in channels.values() for row in rows]
        reasons = []
        if timestamp.year == 1970:
            reasons.append("placeholder_datetime_1970")
        if set(channels) != set(BANDS):
            reasons.append("missing_optical_channel")
        if any(len({row["waveform_sha256"] for row in rows}) != 1
               for rows in channels.values()):
            reasons.append("multiple_distinct_waveforms_at_same_minute")
        if len({(row["sbp"], row["dbp"]) for row in all_rows}) != 1:
            reasons.append("multiple_targets_at_same_minute")
        if len({(row["age"], row["sex"], row["identity_hash"]) for row in all_rows}) != 1:
            reasons.append("inconsistent_identity_or_conditions")
        if any(not np.isfinite(row["waveform"]).all() for row in all_rows):
            reasons.append("nonfinite_waveform")
        if any(np.ptp(row["waveform"]) == 0 for row in all_rows):
            reasons.append("constant_waveform")
        if any(not (np.isfinite(row["sbp"]) and np.isfinite(row["dbp"])
                    and row["sbp"] > row["dbp"] > 0) for row in all_rows):
            reasons.append("nonfinite_or_logically_invalid_target")
        if any(not (np.isfinite(row["age"]) and row["age"] > 0
                    and row["sex"] in (0, 1)) for row in all_rows):
            reasons.append("invalid_condition_metadata")
        if key in reused:
            reasons.append("identical_channel_waveform_reused_at_another_timestamp")
        lineage = [
            dict(band=row["band"], source_file=row["source_file"],
                 csv_row=row["csv_row"], waveform_sha256=row["waveform_sha256"])
            for row in sorted(all_rows, key=lambda x: (x["band"], x["source_file"], x["csv_row"]))
        ]
        if reasons:
            excluded.append(
                dict(record_id=record_id(key), subject=subject,
                     timestamp=timestamp.isoformat(), reasons=reasons,
                     channel_row_counts={band: len(channels.get(band, [])) for band in BANDS},
                     source_rows=lineage)
            )
            continue
        representative = [
            min(channels[band], key=lambda x: (x["source_file"], x["csv_row"]))
            for band in BANDS
        ]
        exact_duplicate_rows_removed += len(all_rows) - len(BANDS)
        first = representative[0]
        kept.append(dict(
            record_id=record_id(key), subject=subject, timestamp=timestamp,
            date=timestamp.date().isoformat(), age=first["age"], sex_code=first["sex"],
            sbp=first["sbp"], dbp=first["dbp"],
            original_splits="|".join(sorted({r["original_split"] for r in all_rows})),
            waveform=np.stack([row["waveform"] for row in representative]),
            channel_hashes=tuple(row["waveform_sha256"] for row in representative),
            source_rows=lineage,
        ))
    for index, row in enumerate(kept):
        row["unit"] = index
    assert kept and all(row["waveform"].shape == (3, 320) for row in kept)
    assert len({(row["subject"], row["timestamp"]) for row in kept}) == len(kept)
    for band_index in range(len(BANDS)):
        assert len({row["channel_hashes"][band_index] for row in kept}) == len(kept)
    return kept, excluded, exact_duplicate_rows_removed


def make_partition(metadata):
    """This function cannot access targets, waveform values, or quality scores."""
    assert set(metadata.columns) == {"unit", "subject", "timestamp", "date"}
    assert metadata.unit.is_unique
    metadata = metadata.sort_values(["subject", "timestamp", "unit"]).reset_index(drop=True)
    test = []
    decisions = []
    for subject, group in metadata.groupby("subject", sort=True):
        dates = sorted(group.date.unique())
        assert len(dates) >= 3, (subject, "Need two development dates plus a test date")
        choices = []
        for boundary in range(2, len(dates)):
            candidate = group.loc[group.date.isin(dates[boundary:]), "unit"].tolist()
            # Closest to 20%, then prefer the smaller test set on an exact tie.
            choices.append((abs(len(candidate) - 0.2 * len(group)),
                            len(candidate), boundary, candidate))
        _, _, boundary, candidate = min(choices, key=lambda item: item[:3])
        test.extend(candidate)
        decisions.append(dict(subject=subject, total_records=len(group),
                              development_dates=dates[:boundary], test_dates=dates[boundary:],
                              test_records=len(candidate),
                              test_fraction=len(candidate) / len(group)))
    test = sorted(test)
    test_set = set(test)
    original_development = metadata.loc[~metadata.unit.isin(test)]
    fixed_embargo = []
    for subject, group in original_development.groupby("subject", sort=True):
        first_test = metadata.loc[
            metadata.subject.eq(subject) & metadata.unit.isin(test), "timestamp"
        ].min()
        assert group.timestamp.max() < first_test
        fixed_embargo.extend(
            group.loc[(first_test - group.timestamp).dt.total_seconds() < EMBARGO_SECONDS,
                      "unit"].tolist()
        )
    development = original_development.loc[~original_development.unit.isin(fixed_embargo)]
    groups_by_subject = []
    for subject, group in development.groupby("subject", sort=True):
        days = [(day, sorted(values.unit.tolist()))
                for day, values in group.groupby("date", sort=True)]
        assert len(days) >= 2, (subject, "Embargo left insufficient development dates")
        groups_by_subject.append((subject, days))
    rng = np.random.RandomState(SEED)
    validation = [[] for _ in range(FOLDS)]
    for subject, days in sorted(
        groups_by_subject, key=lambda item: (-sum(len(g[1]) for g in item[1]), item[0])
    ):
        counts = [0] * FOLDS
        for day, units in sorted(days, key=lambda item: (-len(item[1]), item[0])):
            tie = rng.permutation(FOLDS).tolist()
            fold = min(range(FOLDS),
                       key=lambda f: (counts[f], len(validation[f]), tie.index(f)))
            validation[fold].extend(units)
            counts[fold] += 1
    folds = []
    development_set = set(development.unit)
    for fold in range(FOLDS):
        val = sorted(validation[fold])
        possible_train = development_set - set(val)
        fold_embargo = []
        for subject, group in development.loc[
            development.unit.isin(possible_train)
        ].groupby("subject", sort=True):
            val_times = development.loc[
                development.subject.eq(subject) & development.unit.isin(val), "timestamp"
            ]
            if val_times.empty:
                continue
            val_ns = val_times.astype("datetime64[ns]").astype("int64").to_numpy()
            for row in group.itertuples(index=False):
                min_gap = np.min(np.abs(val_ns - row.timestamp.value)) / 1e9
                if min_gap < EMBARGO_SECONDS:
                    fold_embargo.append(int(row.unit))
        train = sorted(possible_train - set(fold_embargo))
        folds.append(dict(
            fold=fold, units=dict(train=train, validation=val, test=test),
            excluded_from_fold_training_by_embargo=sorted(fold_embargo),
        ))
    return dict(
        development_units=sorted(development_set), fixed_test_units=test,
        excluded_from_development_by_test_embargo=sorted(fixed_embargo),
        subject_boundaries=decisions, folds=folds,
    )
