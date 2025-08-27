# Blocking patterns report

Generated: 2025-08-26T21:02:24.369Z
Scanned files: 1281

## Summary

- Total matches: 884
- medium: 884

## By pattern

- readFileSync: 210
- mkdirSync: 142
- while-infinite: 236
- writeFileSync: 202
- spawnSync: 5
- unlinkSync: 54
- execSync: 35

## Matches (top 200)

- [medium] .gemini_checkpoints/1756182692784-3p2kpi/esbuild.config.js:15 (readFileSync)
  ```
  fs.readFileSync(path.resolve(__dirname, 'package.json'), 'utf8'),
  ```
- [medium] bundle/gemini.js:1759 (readFileSync)
  ```
  return fs.readFileSync("/proc/self/cgroup", "utf8").includes("docker");
  ```
- [medium] bundle/gemini.js:1818 (readFileSync)
  ```
  return fs3.readFileSync("/proc/version", "utf8").toLowerCase().includes("microsoft") ? !isInsideContainer() : false;
  ```
- [medium] bundle/gemini.js:2421 (mkdirSync)
  ```
  fs6.mkdirSync(this.getProjectTempDir(), { recursive: true });
  ```
- [medium] bundle/gemini.js:2500 (readFileSync)
  ```
  const content = readFileSync(filePath, "utf-8");
  ```
- [medium] bundle/gemini.js:16481 (readFileSync)
  ```
  defaultRootsData = fs71.readFileSync(DEFAULT_ROOTS_FILE_PATH);
  ```
- [medium] bundle/gemini.js:18518 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:18643 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:18797 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:23452 (readFileSync)
  ```
  source2 = util3.fs.readFileSync(filename2).toString("utf8");
  ```
- [medium] bundle/gemini.js:28803 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:39263 (readFileSync)
  ```
  return fs71.readFileSync(path87.resolve(process.cwd(), filePath));
  ```
- [medium] bundle/gemini.js:63088 (readFileSync)
  ```
  const installationid = fs8.readFileSync(installationIdFile, "utf-8").trim();
  ```
- [medium] bundle/gemini.js:63096 (mkdirSync)
  ```
  fs8.mkdirSync(dir, { recursive: true });
  ```
- [medium] bundle/gemini.js:63097 (writeFileSync)
  ```
  fs8.writeFileSync(installationIdFile, installationId, "utf-8");
  ```
- [medium] bundle/gemini.js:63273 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:78479 (spawnSync)
  ```
  const result = cp.spawnSync(parsed.command, parsed.args, parsed.options);
  ```
- [medium] bundle/gemini.js:78657 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:80217 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:101095 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:101114 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:102590 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:104597 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:107401 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:107761 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:108522 (while-infinite)
  ```
  } while (true);
  ```
- [medium] bundle/gemini.js:109010 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:109063 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:111711 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:111956 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:121450 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:126183 (readFileSync)
  ```
  originalContent = fileExists ? fs19.readFileSync(file_path, "utf8") : "";
  ```
- [medium] bundle/gemini.js:126188 (mkdirSync)
  ```
  fs19.mkdirSync(dirName, { recursive: true });
  ```
- [medium] bundle/gemini.js:132226 (mkdirSync)
  ```
  fs71.mkdirSync(traceDir, { recursive: true });
  ```
- [medium] bundle/gemini.js:132461 (writeFileSync)
  ```
  fs71.writeFileSync(legendPath, JSON.stringify(legend));
  ```
- [medium] bundle/gemini.js:135659 (mkdirSync)
  ```
  _fs.mkdirSync(directoryName);
  ```
- [medium] bundle/gemini.js:135797 (mkdirSync)
  ```
  _fs.mkdirSync(_path.dirname(profilePath), { recursive: true });
  ```
- [medium] bundle/gemini.js:135800 (writeFileSync)
  ```
  _fs.writeFileSync(profilePath, JSON.stringify(cleanupPaths(profile)));
  ```
- [medium] bundle/gemini.js:135858 (readFileSync)
  ```
  buffer = _fs.readFileSync(fileName);
  ```
- [medium] bundle/gemini.js:136021 (unlinkSync)
  ```
  return _fs.unlinkSync(path87);
  ```
- [medium] bundle/gemini.js:136433 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:139026 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:139455 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:139654 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:139692 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:139981 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:140027 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:140559 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:140619 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:140667 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:140680 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:140869 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:141023 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:141122 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:141201 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:141300 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:141510 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:143587 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:145162 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:145238 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:145334 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:146318 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:148702 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:150279 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:159711 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:160062 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:162939 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:163337 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:164902 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:165229 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:165545 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:165618 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:166388 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:168130 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:168228 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:168393 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:168916 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:169367 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:171376 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:176683 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:180011 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:191156 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:196970 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:199207 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:203125 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:204896 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:204966 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:205071 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:207167 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:210280 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:217823 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:257917 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:257959 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:261613 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:263909 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:264746 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:268932 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:271998 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:272573 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:272845 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:274389 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:275627 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:275835 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:276521 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:277099 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:277557 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:283097 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:284600 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:286190 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:287621 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:299180 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:310777 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:315692 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:319767 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:323673 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:327516 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:339143 (mkdirSync)
  ```
  return opts.mkdirSync(path87, opts);
  ```
- [medium] bundle/gemini.js:339153 (mkdirSync)
  ```
  opts.mkdirSync(path87, opts);
  ```
- [medium] bundle/gemini.js:339258 (mkdirSync)
  ```
  return opts.mkdirSync(path87, opts);
  ```
- [medium] bundle/gemini.js:339262 (mkdirSync)
  ```
  opts.mkdirSync(path87, opts);
  ```
- [medium] bundle/gemini.js:340437 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:371159 (readFileSync)
  ```
  readFileSync(_filePath, _encoding) {
  ```
- [medium] bundle/gemini.js:371165 (writeFileSync)
  ```
  writeFileSync(_filePath, _fileText) {
  ```
- [medium] bundle/gemini.js:371171 (mkdirSync)
  ```
  mkdirSync(_dirPath) {
  ```
- [medium] bundle/gemini.js:371267 (readFileSync)
  ```
  readFileSync(filePath, encoding = "utf-8") {
  ```
- [medium] bundle/gemini.js:371268 (readFileSync)
  ```
  return fs__namespace.readFileSync(filePath, encoding);
  ```
- [medium] bundle/gemini.js:371280 (writeFileSync)
  ```
  writeFileSync(filePath, fileText) {
  ```
- [medium] bundle/gemini.js:371281 (writeFileSync)
  ```
  fs__namespace.writeFileSync(filePath, fileText);
  ```
- [medium] bundle/gemini.js:371286 (mkdirSync)
  ```
  mkdirSync(dirPath) {
  ```
- [medium] bundle/gemini.js:371400 (readFileSync)
  ```
  return transactionalFileSystem.readFileSync(standardizedPath, encoding);
  ```
- [medium] bundle/gemini.js:371426 (writeFileSync)
  ```
  transactionalFileSystem.writeFileSync(filePath, writeByteOrderMark ? "\uFEFF" + data : data);
  ```
- [medium] bundle/gemini.js:371555 (readFileSync)
  ```
  return fileSystem.readFileSync(filePath, encoding);
  ```
- [medium] bundle/gemini.js:371711 (readFileSync)
  ```
  return Promise.resolve(this.readFileSync(filePath, encoding));
  ```
- [medium] bundle/gemini.js:371716 (readFileSync)
  ```
  readFileSync(filePath, encoding = "utf-8") {
  ```
- [medium] bundle/gemini.js:371727 (writeFileSync)
  ```
  this.writeFileSync(filePath, fileText);
  ```
- [medium] bundle/gemini.js:371730 (writeFileSync)
  ```
  writeFileSync(filePath, fileText) {
  ```
- [medium] bundle/gemini.js:371731 (writeFileSync)
  ```
  this.#writeFileSync(filePath, fileText);
  ```
- [medium] bundle/gemini.js:371733 (writeFileSync)
  ```
  #writeFileSync(filePath, fileText) {
  ```
- [medium] bundle/gemini.js:371739 (mkdirSync)
  ```
  this.mkdirSync(dirPath);
  ```
- [medium] bundle/gemini.js:371742 (mkdirSync)
  ```
  mkdirSync(dirPath) {
  ```
- [medium] bundle/gemini.js:371753 (readFileSync)
  ```
  const fileText = this.readFileSync(standardizedSrcPath);
  ```
- [medium] bundle/gemini.js:371755 (writeFileSync)
  ```
  this.writeFileSync(standardizedDestPath, fileText);
  ```
- [medium] bundle/gemini.js:371778 (readFileSync)
  ```
  this.writeFileSync(standardizedDestPath, this.readFileSync(standardizedSrcPath));
  ```
- [medium] bundle/gemini.js:371778 (writeFileSync)
  ```
  this.writeFileSync(standardizedDestPath, this.readFileSync(standardizedSrcPath));
  ```
- [medium] bundle/gemini.js:371897 (readFileSync)
  ```
  readFileSync(filePath, encoding = "utf-8") {
  ```
- [medium] bundle/gemini.js:371899 (readFileSync)
  ```
  return fs71.readFileSync(filePath, encoding);
  ```
- [medium] bundle/gemini.js:371907 (writeFileSync)
  ```
  writeFileSync(filePath, fileText) {
  ```
- [medium] bundle/gemini.js:371908 (writeFileSync)
  ```
  fs71.writeFileSync(filePath, fileText);
  ```
- [medium] bundle/gemini.js:371913 (mkdirSync)
  ```
  mkdirSync(dirPath) {
  ```
- [medium] bundle/gemini.js:371914 (mkdirSync)
  ```
  fs71.mkdirSync(dirPath);
  ```
- [medium] bundle/gemini.js:372271 (mkdirSync)
  ```
  this.#fileSystem.mkdirSync(operation.dir.path);
  ```
- [medium] bundle/gemini.js:372296 (writeFileSync)
  ```
  this.writeFileSync(newFilePath, fileText);
  ```
- [medium] bundle/gemini.js:372387 (mkdirSync)
  ```
  this.#fileSystem.mkdirSync(dirPath);
  ```
- [medium] bundle/gemini.js:372455 (readFileSync)
  ```
  return this.readFileSync(filePath, encoding);
  ```
- [medium] bundle/gemini.js:372463 (readFileSync)
  ```
  readFileSync(filePath, encoding) {
  ```
- [medium] bundle/gemini.js:372468 (readFileSync)
  ```
  return this.#fileSystem.readFileSync(filePath, encoding);
  ```
- [medium] bundle/gemini.js:372573 (writeFileSync)
  ```
  writeFileSync(filePath, fileText) {
  ```
- [medium] bundle/gemini.js:372579 (writeFileSync)
  ```
  this.#fileSystem.writeFileSync(filePath, fileText);
  ```
- [medium] bundle/gemini.js:372681 (mkdirSync)
  ```
  this.#fileSystem.mkdirSync(dir.path);
  ```
- [medium] bundle/gemini.js:372748 (readFileSync)
  ```
  return transactionalFileSystem.readFileSync(filePath, getEncoding());
  ```
- [medium] bundle/gemini.js:372974 (readFileSync)
  ```
  readFile: (path88) => fileSystemWrapper.readFileSync(fileSystemWrapper.getStandardizedAbsolutePath(path88), options2.encoding),
  ```
- [medium] bundle/gemini.js:373029 (readFileSync)
  ```
  const text = this.#fileSystem.readFileSync(this.#tsConfigFilePath, this.#encoding);
  ```
- [medium] bundle/gemini.js:377386 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:387055 (writeFileSync)
  ```
  this._context.fileSystemWrapper.writeFileSync(this.getFilePath(), this.#getTextForSave());
  ```
- [medium] bundle/gemini.js:391015 (writeFileSync)
  ```
  fileSystem.writeFileSync(file.filePath, file.writeByteOrderMark ? "\uFEFF" + file.text : file.text);
  ```
- [medium] bundle/gemini.js:392004 (writeFileSync)
  ```
  fileSystemWrapper.writeFileSync(emitResult.filePath, emitResult.fileText);
  ```
- [medium] bundle/gemini.js:395874 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:396651 (mkdirSync)
  ```
  fs22.mkdirSync(dirName, { recursive: true });
  ```
- [medium] bundle/gemini.js:402593 (readFileSync)
  ```
  return (0, exports2.detect)(fs71.readFileSync(filepath));
  ```
- [medium] bundle/gemini.js:402623 (execSync)
  ```
  const output = execSync("chcp", { encoding: "utf8" });
  ```
- [medium] bundle/gemini.js:402641 (execSync)
  ```
  locale = execSync("locale charmap", { encoding: "utf8" }).toString().trim();
  ```
- [medium] bundle/gemini.js:406871 (readFileSync)
  ```
  const pgrepLines = fs23.readFileSync(tempFilePath, "utf8").split(EOL3).filter(Boolean);
  ```
- [medium] bundle/gemini.js:406942 (unlinkSync)
  ```
  fs23.unlinkSync(tempFilePath);
  ```
- [medium] bundle/gemini.js:412755 (readFileSync)
  ```
  const original = fs24.readFileSync(filePath, "utf-8");
  ```
- [medium] bundle/gemini.js:412842 (readFileSync)
  ```
  const originalContent = fs25.readFileSync(this.params.file_path, "utf-8");
  ```
- [medium] bundle/gemini.js:412855 (writeFileSync)
  ```
  fs25.writeFileSync(this.params.file_path, newContent, "utf-8");
  ```
- [medium] bundle/gemini.js:412899 (writeFileSync)
  ```
  fs25.writeFileSync(this.params.file_path, newContent, "utf-8");
  ```
- [medium] bundle/gemini.js:412926 (readFileSync)
  ```
  const originalContent = fs25.readFileSync(this.params.file_path, "utf-8");
  ```
- [medium] bundle/gemini.js:412957 (writeFileSync)
  ```
  fs25.writeFileSync(this.params.file_path, newContent, "utf-8");
  ```
- [medium] bundle/gemini.js:414303 (readFileSync)
  ```
  const basePrompt = systemMdEnabled ? fs28.readFileSync(systemMdPath, "utf8") : `
  ```
- [medium] bundle/gemini.js:414524 (mkdirSync)
  ```
  fs28.mkdirSync(path30.dirname(systemMdPath), { recursive: true });
  ```
- [medium] bundle/gemini.js:414525 (writeFileSync)
  ```
  fs28.writeFileSync(systemMdPath, basePrompt);
  ```
- [medium] bundle/gemini.js:414534 (mkdirSync)
  ```
  fs28.mkdirSync(path30.dirname(resolvedPath), { recursive: true });
  ```
- [medium] bundle/gemini.js:414535 (writeFileSync)
  ```
  fs28.writeFileSync(resolvedPath, basePrompt);
  ```
- [medium] bundle/gemini.js:417240 (readFileSync)
  ```
  content = fs29.readFileSync(patternsFilePath, "utf-8");
  ```
- [medium] bundle/gemini.js:426194 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:426427 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:428316 (readFileSync)
  ```
  ignorer.add(fs37.readFileSync(gitignorePath, "utf8"));
  ```
- [medium] bundle/gemini.js:428322 (readFileSync)
  ```
  ignorer.add(fs37.readFileSync(geminiignorePath, "utf8"));
  ```
- [medium] bundle/gemini.js:429355 (while-infinite)
  ```
  while (true) {
  ```
- [medium] bundle/gemini.js:429904 (mkdirSync)
  ```
  fs39.mkdirSync(chatsDir, { recursive: true });
  ```
- [medium] bundle/gemini.js:430067 (readFileSync)
  ```
  this.cachedLastConvData = fs39.readFileSync(this.conversationFile, "utf8");
  ```
- [medium] bundle/gemini.js:430096 (writeFileSync)
  ```
  fs39.writeFileSync(this.conversationFile, newContent);
  ```
- [medium] bundle/gemini.js:430119 (unlinkSync)
  ```
  fs39.unlinkSync(sessionPath);
  ```
- [medium] bundle/gemini.js:430147 (execSync)
  ```
  const result = child_process.execSync(`where.exe ${VSCODE_COMMAND}`).toString().trim();
  ```
- [medium] bundle/gemini.js:430153 (execSync)
  ```
  child_process.execSync(`command -v ${VSCODE_COMMAND}`, {
  ```
- [medium] bundle/gemini.js:430207 (execSync)
  ```
  child_process.execSync(command, { stdio: "pipe" });
  ```
- [medium] bundle/gemini.js:430339 (mkdirSync)
  ```
  fs41.mkdirSync(diffDir, { recursive: true });
  ```
- [medium] bundle/gemini.js:430346 (writeFileSync)
  ```
  fs41.writeFileSync(tempOldPath, currentContent, "utf8");
  ```
- [medium] bundle/gemini.js:430347 (writeFileSync)
  ```
  fs41.writeFileSync(tempNewPath, proposedContent, "utf8");
  ```

...and 759 more matches. See JSON report for full list.
