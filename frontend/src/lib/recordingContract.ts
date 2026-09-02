const DATASET_REPO_COMPONENT = /^[A-Za-z0-9._-]+$/;

export const isExactDatasetRepoComponent = (
  value: unknown,
): value is string =>
  typeof value === "string" &&
  value.length > 0 &&
  value.trim() === value &&
  value !== "." &&
  value !== ".." &&
  DATASET_REPO_COMPONENT.test(value);

export const isExactDatasetRepoId = (value: unknown): value is string => {
  if (typeof value !== "string" || value.trim() !== value) return false;
  const parts = value.split("/");
  return (
    parts.length === 2 &&
    parts.every(isExactDatasetRepoComponent) &&
    !parts[1].startsWith("eval_")
  );
};

export const makeStadiaDatasetRepoId = (
  username: unknown,
  datasetName: unknown,
): string | null => {
  if (
    !isExactDatasetRepoComponent(username) ||
    !isExactDatasetRepoComponent(datasetName)
  ) {
    return null;
  }
  const result = `${username}/${datasetName}`;
  return isExactDatasetRepoId(result) ? result : null;
};
