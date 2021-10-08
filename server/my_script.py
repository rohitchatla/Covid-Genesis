print("hello from shell")

'''
//1
  //   let options = {
  //     mode: "text",
  //     //pythonPath: "path/to/python",
  //     pythonOptions: ["-u"], // get print results in real-time
  //     //scriptPath: "path/to/my/scripts",
  //     //args: ["value1", "value2", "value3"],
  //   };
  //   PythonShell.run("my_script.py", options, function (err, results) {
  //     //null
  //     if (err) throw err;
  //     console.log("finished");
  //     console.log("results: %j", results);
  //   });
  //2
  //   let pyshell = new PythonShell("my_script.py");
  //   // sends a message to the Python script via stdin
  //   pyshell.send("hello");
  //   pyshell.on("message", function (message) {
  //     // received a message sent from the Python script (a simple "print" statement)
  //     console.log(message);
  //   });
  //   // end the input stream and allow the process to exit
  //   pyshell.end(function (err, code, signal) {
  //     if (err) throw err;
  //     console.log("The exit code was: " + code);
  //     console.log("The exit signal was: " + signal);
  //     console.log("finished");
  //   });
  //3
  // send a message in text mode
  //   let shell = new PythonShell("my_script.py", { mode: "text" });
  //   shell.send("hello world!");
  //4
  // send a message in JSON mode
  //   let shell = new PythonShell("my_script.py", { mode: "json" });
  //   shell.send({ command: "do_stuff", args: [1, 2, 3] });
'''