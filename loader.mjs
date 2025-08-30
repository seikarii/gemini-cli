
import { resolve as resolveTs } from 'ts-node/esm';
import { createMatchPath, loadConfig } from 'tsconfig-paths';
import { pathToFileURL } from 'url';

const { absoluteBaseUrl, paths } = loadConfig();
const matchPath = createMatchPath(absoluteBaseUrl, paths);

export function resolve(specifier, context, defaultResolve) {
  const match = matchPath(specifier);
  if (match) {
    return resolveTs(
      pathToFileURL(match).href,
      context,
      defaultResolve
    );
  }
  return resolveTs(specifier, context, defaultResolve);
}

export { load, transformSource } from 'ts-node/esm';
