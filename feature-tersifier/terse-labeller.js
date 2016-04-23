
"use strict";

var fs = require('fs');
var lineReader = require('readline');

var lineCount = 0;

lineReader.createInterface({
    input: fs.createReadStream('feature-base-words.txt')
})
.on('line', function(line) {
    process.stdout.write(line + " " + toLetters(++lineCount).toLowerCase() + "\n");
});

// source: http://stackoverflow.com/a/11090169

function toLetters(num) {
    "use strict";
    var mod = num % 26,
        pow = num / 26 | 0,
        out = mod ? String.fromCharCode(64 + mod) : (--pow, 'Z');
    return pow ? toLetters(pow) + out : out;
}

function fromLetters(str) {
    "use strict";
    var out = 0, len = str.length, pos = len;
    while (--pos > -1) {
        out += (str.charCodeAt(pos) - 64) * Math.pow(26, len - 1 - pos);
    }
    return out;
}




