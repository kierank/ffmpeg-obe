/*
 * Various utilities for command line tools
 * Copyright (c) 2000-2003 Fabrice Bellard
 *
 * This file is part of Libav.
 *
 * Libav is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * Libav is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Libav; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>

/* Include only the enabled headers since some compilers (namely, Sun
   Studio) will not omit unused inline functions and create undefined
   references to libraries that are not being built. */

#include "config.h"
#include "libavformat/avformat.h"
#include "libavfilter/avfilter.h"
#include "libavdevice/avdevice.h"
#include "libavresample/avresample.h"
#include "libswscale/swscale.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/mathematics.h"
#include "libavutil/imgutils.h"
#include "libavutil/parseutils.h"
#include "libavutil/pixdesc.h"
#include "libavutil/eval.h"
#include "libavutil/dict.h"
#include "libavutil/opt.h"
#include "cmdutils.h"
#include "version.h"
#if CONFIG_NETWORK
#include "libavformat/network.h"
#endif
#if HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif

struct SwsContext *sws_opts;
AVDictionary *format_opts, *codec_opts;

static const int this_year = 2012;

void init_opts(void)
{
#if CONFIG_SWSCALE
    sws_opts = sws_getContext(16, 16, 0, 16, 16, 0, SWS_BICUBIC,
                              NULL, NULL, NULL);
#endif
}

void uninit_opts(void)
{
#if CONFIG_SWSCALE
    sws_freeContext(sws_opts);
    sws_opts = NULL;
#endif
    av_dict_free(&format_opts);
    av_dict_free(&codec_opts);
}

void log_callback_help(void *ptr, int level, const char *fmt, va_list vl)
{
    vfprintf(stdout, fmt, vl);
}

double parse_number_or_die(const char *context, const char *numstr, int type,
                           double min, double max)
{
    char *tail;
    const char *error;
    double d = av_strtod(numstr, &tail);
    if (*tail)
        error = "Expected number for %s but found: %s\n";
    else if (d < min || d > max)
        error = "The value for %s was %s which is not within %f - %f\n";
    else if (type == OPT_INT64 && (int64_t)d != d)
        error = "Expected int64 for %s but found %s\n";
    else if (type == OPT_INT && (int)d != d)
        error = "Expected int for %s but found %s\n";
    else
        return d;
    av_log(NULL, AV_LOG_FATAL, error, context, numstr, min, max);
    exit_program(1);
    return 0;
}

int64_t parse_time_or_die(const char *context, const char *timestr,
                          int is_duration)
{
    int64_t us;
    if (av_parse_time(&us, timestr, is_duration) < 0) {
        av_log(NULL, AV_LOG_FATAL, "Invalid %s specification for %s: %s\n",
               is_duration ? "duration" : "date", context, timestr);
        exit_program(1);
    }
    return us;
}

void show_help_options(const OptionDef *options, const char *msg, int req_flags,
                       int rej_flags, int alt_flags)
{
    const OptionDef *po;
    int first;

    first = 1;
    for (po = options; po->name != NULL; po++) {
        char buf[64];

        if (((po->flags & req_flags) != req_flags) ||
            (alt_flags && !(po->flags & alt_flags)) ||
            (po->flags & rej_flags))
            continue;

        if (first) {
            printf("%s\n", msg);
            first = 0;
        }
        av_strlcpy(buf, po->name, sizeof(buf));
        if (po->argname) {
            av_strlcat(buf, " ", sizeof(buf));
            av_strlcat(buf, po->argname, sizeof(buf));
        }
        printf("-%-17s  %s\n", buf, po->help);
    }
    printf("\n");
}

void show_help_children(const AVClass *class, int flags)
{
    const AVClass *child = NULL;
    av_opt_show2(&class, NULL, flags, 0);
    printf("\n");

    while (child = av_opt_child_class_next(class, child))
        show_help_children(child, flags);
}

static const OptionDef *find_option(const OptionDef *po, const char *name)
{
    const char *p = strchr(name, ':');
    int len = p ? p - name : strlen(name);

    while (po->name != NULL) {
        if (!strncmp(name, po->name, len) && strlen(po->name) == len)
            break;
        po++;
    }
    return po;
}

#if defined(_WIN32) && !defined(__MINGW32CE__)
#include <windows.h>
#include <shellapi.h>
/* Will be leaked on exit */
static char** win32_argv_utf8 = NULL;
static int win32_argc = 0;

/**
 * Prepare command line arguments for executable.
 * For Windows - perform wide-char to UTF-8 conversion.
 * Input arguments should be main() function arguments.
 * @param argc_ptr Arguments number (including executable)
 * @param argv_ptr Arguments list.
 */
static void prepare_app_arguments(int *argc_ptr, char ***argv_ptr)
{
    char *argstr_flat;
    wchar_t **argv_w;
    int i, buffsize = 0, offset = 0;

    if (win32_argv_utf8) {
        *argc_ptr = win32_argc;
        *argv_ptr = win32_argv_utf8;
        return;
    }

    win32_argc = 0;
    argv_w = CommandLineToArgvW(GetCommandLineW(), &win32_argc);
    if (win32_argc <= 0 || !argv_w)
        return;

    /* determine the UTF-8 buffer size (including NULL-termination symbols) */
    for (i = 0; i < win32_argc; i++)
        buffsize += WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1,
                                        NULL, 0, NULL, NULL);

    win32_argv_utf8 = av_mallocz(sizeof(char *) * (win32_argc + 1) + buffsize);
    argstr_flat     = (char *)win32_argv_utf8 + sizeof(char *) * (win32_argc + 1);
    if (win32_argv_utf8 == NULL) {
        LocalFree(argv_w);
        return;
    }

    for (i = 0; i < win32_argc; i++) {
        win32_argv_utf8[i] = &argstr_flat[offset];
        offset += WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1,
                                      &argstr_flat[offset],
                                      buffsize - offset, NULL, NULL);
    }
    win32_argv_utf8[i] = NULL;
    LocalFree(argv_w);

    *argc_ptr = win32_argc;
    *argv_ptr = win32_argv_utf8;
}
#else
static inline void prepare_app_arguments(int *argc_ptr, char ***argv_ptr)
{
    /* nothing to do */
}
#endif /* WIN32 && !__MINGW32CE__ */

int parse_option(void *optctx, const char *opt, const char *arg,
                 const OptionDef *options)
{
    const OptionDef *po;
    int bool_val = 1;
    int *dstcount;
    void *dst;

    po = find_option(options, opt);
    if (!po->name && opt[0] == 'n' && opt[1] == 'o') {
        /* handle 'no' bool option */
        po = find_option(options, opt + 2);
        if ((po->name && (po->flags & OPT_BOOL)))
            bool_val = 0;
    }
    if (!po->name)
        po = find_option(options, "default");
    if (!po->name) {
        av_log(NULL, AV_LOG_ERROR, "Unrecognized option '%s'\n", opt);
        return AVERROR(EINVAL);
    }
    if (po->flags & HAS_ARG && !arg) {
        av_log(NULL, AV_LOG_ERROR, "Missing argument for option '%s'\n", opt);
        return AVERROR(EINVAL);
    }

    /* new-style options contain an offset into optctx, old-style address of
     * a global var*/
    dst = po->flags & (OPT_OFFSET | OPT_SPEC) ? (uint8_t *)optctx + po->u.off
                                              : po->u.dst_ptr;

    if (po->flags & OPT_SPEC) {
        SpecifierOpt **so = dst;
        char *p = strchr(opt, ':');

        dstcount = (int *)(so + 1);
        *so = grow_array(*so, sizeof(**so), dstcount, *dstcount + 1);
        (*so)[*dstcount - 1].specifier = av_strdup(p ? p + 1 : "");
        dst = &(*so)[*dstcount - 1].u;
    }

    if (po->flags & OPT_STRING) {
        char *str;
        str = av_strdup(arg);
        av_freep(dst);
        *(char **)dst = str;
    } else if (po->flags & OPT_BOOL) {
        *(int *)dst = bool_val;
    } else if (po->flags & OPT_INT) {
        *(int *)dst = parse_number_or_die(opt, arg, OPT_INT64, INT_MIN, INT_MAX);
    } else if (po->flags & OPT_INT64) {
        *(int64_t *)dst = parse_number_or_die(opt, arg, OPT_INT64, INT64_MIN, INT64_MAX);
    } else if (po->flags & OPT_TIME) {
        *(int64_t *)dst = parse_time_or_die(opt, arg, 1);
    } else if (po->flags & OPT_FLOAT) {
        *(float *)dst = parse_number_or_die(opt, arg, OPT_FLOAT, -INFINITY, INFINITY);
    } else if (po->flags & OPT_DOUBLE) {
        *(double *)dst = parse_number_or_die(opt, arg, OPT_DOUBLE, -INFINITY, INFINITY);
    } else if (po->u.func_arg) {
        int ret = po->flags & OPT_FUNC2 ? po->u.func2_arg(optctx, opt, arg)
                                        : po->u.func_arg(opt, arg);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR,
                   "Failed to set value '%s' for option '%s'\n", arg, opt);
            return ret;
        }
    }
    if (po->flags & OPT_EXIT)
        exit_program(0);
    return !!(po->flags & HAS_ARG);
}

void parse_options(void *optctx, int argc, char **argv, const OptionDef *options,
                   void (*parse_arg_function)(void *, const char*))
{
    const char *opt;
    int optindex, handleoptions = 1, ret;

    /* perform system-dependent conversions for arguments list */
    prepare_app_arguments(&argc, &argv);

    /* parse options */
    optindex = 1;
    while (optindex < argc) {
        opt = argv[optindex++];

        if (handleoptions && opt[0] == '-' && opt[1] != '\0') {
            if (opt[1] == '-' && opt[2] == '\0') {
                handleoptions = 0;
                continue;
            }
            opt++;

            if ((ret = parse_option(optctx, opt, argv[optindex], options)) < 0)
                exit_program(1);
            optindex += ret;
        } else {
            if (parse_arg_function)
                parse_arg_function(optctx, opt);
        }
    }
}

int locate_option(int argc, char **argv, const OptionDef *options,
                  const char *optname)
{
    const OptionDef *po;
    int i;

    for (i = 1; i < argc; i++) {
        const char *cur_opt = argv[i];

        if (*cur_opt++ != '-')
            continue;

        po = find_option(options, cur_opt);
        if (!po->name && cur_opt[0] == 'n' && cur_opt[1] == 'o')
            po = find_option(options, cur_opt + 2);

        if ((!po->name && !strcmp(cur_opt, optname)) ||
             (po->name && !strcmp(optname, po->name)))
            return i;

        if (!po || po->flags & HAS_ARG)
            i++;
    }
    return 0;
}

void parse_loglevel(int argc, char **argv, const OptionDef *options)
{
    int idx = locate_option(argc, argv, options, "loglevel");
    if (!idx)
        idx = locate_option(argc, argv, options, "v");
    if (idx && argv[idx + 1])
        opt_loglevel("loglevel", argv[idx + 1]);
}

#define FLAGS (o->type == AV_OPT_TYPE_FLAGS) ? AV_DICT_APPEND : 0
int opt_default(const char *opt, const char *arg)
{
    const AVOption *o;
    char opt_stripped[128];
    const char *p;
    const AVClass *cc = avcodec_get_class(), *fc = avformat_get_class(), *sc = sws_get_class();

    if (!(p = strchr(opt, ':')))
        p = opt + strlen(opt);
    av_strlcpy(opt_stripped, opt, FFMIN(sizeof(opt_stripped), p - opt + 1));

    if ((o = av_opt_find(&cc, opt_stripped, NULL, 0,
                         AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ)) ||
        ((opt[0] == 'v' || opt[0] == 'a' || opt[0] == 's') &&
         (o = av_opt_find(&cc, opt + 1, NULL, 0, AV_OPT_SEARCH_FAKE_OBJ))))
        av_dict_set(&codec_opts, opt, arg, FLAGS);
    else if ((o = av_opt_find(&fc, opt, NULL, 0,
                              AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ)))
        av_dict_set(&format_opts, opt, arg, FLAGS);
    else if ((o = av_opt_find(&sc, opt, NULL, 0,
                              AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ))) {
        // XXX we only support sws_flags, not arbitrary sws options
        int ret = av_opt_set(sws_opts, opt, arg, 0);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error setting option %s.\n", opt);
            return ret;
        }
    }

    if (o)
        return 0;
    av_log(NULL, AV_LOG_ERROR, "Unrecognized option '%s'\n", opt);
    return AVERROR_OPTION_NOT_FOUND;
}

int opt_loglevel(const char *opt, const char *arg)
{
    const struct { const char *name; int level; } log_levels[] = {
        { "quiet"  , AV_LOG_QUIET   },
        { "panic"  , AV_LOG_PANIC   },
        { "fatal"  , AV_LOG_FATAL   },
        { "error"  , AV_LOG_ERROR   },
        { "warning", AV_LOG_WARNING },
        { "info"   , AV_LOG_INFO    },
        { "verbose", AV_LOG_VERBOSE },
        { "debug"  , AV_LOG_DEBUG   },
    };
    char *tail;
    int level;
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(log_levels); i++) {
        if (!strcmp(log_levels[i].name, arg)) {
            av_log_set_level(log_levels[i].level);
            return 0;
        }
    }

    level = strtol(arg, &tail, 10);
    if (*tail) {
        av_log(NULL, AV_LOG_FATAL, "Invalid loglevel \"%s\". "
               "Possible levels are numbers or:\n", arg);
        for (i = 0; i < FF_ARRAY_ELEMS(log_levels); i++)
            av_log(NULL, AV_LOG_FATAL, "\"%s\"\n", log_levels[i].name);
        exit_program(1);
    }
    av_log_set_level(level);
    return 0;
}

int opt_timelimit(const char *opt, const char *arg)
{
#if HAVE_SETRLIMIT
    int lim = parse_number_or_die(opt, arg, OPT_INT64, 0, INT_MAX);
    struct rlimit rl = { lim, lim + 1 };
    if (setrlimit(RLIMIT_CPU, &rl))
        perror("setrlimit");
#else
    av_log(NULL, AV_LOG_WARNING, "-%s not implemented on this OS\n", opt);
#endif
    return 0;
}

void print_error(const char *filename, int err)
{
    char errbuf[128];
    const char *errbuf_ptr = errbuf;

    if (av_strerror(err, errbuf, sizeof(errbuf)) < 0)
        errbuf_ptr = strerror(AVUNERROR(err));
    av_log(NULL, AV_LOG_ERROR, "%s: %s\n", filename, errbuf_ptr);
}

static int warned_cfg = 0;

#define INDENT        1
#define SHOW_VERSION  2
#define SHOW_CONFIG   4

#define PRINT_LIB_INFO(libname, LIBNAME, flags, level)                  \
    if (CONFIG_##LIBNAME) {                                             \
        const char *indent = flags & INDENT? "  " : "";                 \
        if (flags & SHOW_VERSION) {                                     \
            unsigned int version = libname##_version();                 \
            av_log(NULL, level,                                         \
                   "%slib%-10s %2d.%3d.%2d / %2d.%3d.%2d\n",            \
                   indent, #libname,                                    \
                   LIB##LIBNAME##_VERSION_MAJOR,                        \
                   LIB##LIBNAME##_VERSION_MINOR,                        \
                   LIB##LIBNAME##_VERSION_MICRO,                        \
                   version >> 16, version >> 8 & 0xff, version & 0xff); \
        }                                                               \
        if (flags & SHOW_CONFIG) {                                      \
            const char *cfg = libname##_configuration();                \
            if (strcmp(LIBAV_CONFIGURATION, cfg)) {                     \
                if (!warned_cfg) {                                      \
                    av_log(NULL, level,                                 \
                            "%sWARNING: library configuration mismatch\n", \
                            indent);                                    \
                    warned_cfg = 1;                                     \
                }                                                       \
                av_log(NULL, level, "%s%-11s configuration: %s\n",      \
                        indent, #libname, cfg);                         \
            }                                                           \
        }                                                               \
    }                                                                   \

static void print_all_libs_info(int flags, int level)
{
    PRINT_LIB_INFO(avutil,   AVUTIL,   flags, level);
    PRINT_LIB_INFO(avcodec,  AVCODEC,  flags, level);
    PRINT_LIB_INFO(avformat, AVFORMAT, flags, level);
    PRINT_LIB_INFO(avdevice, AVDEVICE, flags, level);
    PRINT_LIB_INFO(avfilter, AVFILTER, flags, level);
    PRINT_LIB_INFO(avresample, AVRESAMPLE, flags, level);
    PRINT_LIB_INFO(swscale,  SWSCALE,  flags, level);
}

void show_banner(void)
{
    av_log(NULL, AV_LOG_INFO,
           "%s version " LIBAV_VERSION ", Copyright (c) %d-%d the Libav developers\n",
           program_name, program_birth_year, this_year);
    av_log(NULL, AV_LOG_INFO, "  built on %s %s with %s\n",
           __DATE__, __TIME__, CC_IDENT);
    av_log(NULL, AV_LOG_VERBOSE, "  configuration: " LIBAV_CONFIGURATION "\n");
    print_all_libs_info(INDENT|SHOW_CONFIG,  AV_LOG_VERBOSE);
    print_all_libs_info(INDENT|SHOW_VERSION, AV_LOG_VERBOSE);
}

int show_version(const char *opt, const char *arg)
{
    av_log_set_callback(log_callback_help);
    printf("%s " LIBAV_VERSION "\n", program_name);
    print_all_libs_info(SHOW_VERSION, AV_LOG_INFO);

    return 0;
}

int show_license(const char *opt, const char *arg)
{
    printf(
#if CONFIG_NONFREE
    "This version of %s has nonfree parts compiled in.\n"
    "Therefore it is not legally redistributable.\n",
    program_name
#elif CONFIG_GPLV3
    "%s is free software; you can redistribute it and/or modify\n"
    "it under the terms of the GNU General Public License as published by\n"
    "the Free Software Foundation; either version 3 of the License, or\n"
    "(at your option) any later version.\n"
    "\n"
    "%s is distributed in the hope that it will be useful,\n"
    "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
    "GNU General Public License for more details.\n"
    "\n"
    "You should have received a copy of the GNU General Public License\n"
    "along with %s.  If not, see <http://www.gnu.org/licenses/>.\n",
    program_name, program_name, program_name
#elif CONFIG_GPL
    "%s is free software; you can redistribute it and/or modify\n"
    "it under the terms of the GNU General Public License as published by\n"
    "the Free Software Foundation; either version 2 of the License, or\n"
    "(at your option) any later version.\n"
    "\n"
    "%s is distributed in the hope that it will be useful,\n"
    "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
    "GNU General Public License for more details.\n"
    "\n"
    "You should have received a copy of the GNU General Public License\n"
    "along with %s; if not, write to the Free Software\n"
    "Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA\n",
    program_name, program_name, program_name
#elif CONFIG_LGPLV3
    "%s is free software; you can redistribute it and/or modify\n"
    "it under the terms of the GNU Lesser General Public License as published by\n"
    "the Free Software Foundation; either version 3 of the License, or\n"
    "(at your option) any later version.\n"
    "\n"
    "%s is distributed in the hope that it will be useful,\n"
    "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
    "GNU Lesser General Public License for more details.\n"
    "\n"
    "You should have received a copy of the GNU Lesser General Public License\n"
    "along with %s.  If not, see <http://www.gnu.org/licenses/>.\n",
    program_name, program_name, program_name
#else
    "%s is free software; you can redistribute it and/or\n"
    "modify it under the terms of the GNU Lesser General Public\n"
    "License as published by the Free Software Foundation; either\n"
    "version 2.1 of the License, or (at your option) any later version.\n"
    "\n"
    "%s is distributed in the hope that it will be useful,\n"
    "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU\n"
    "Lesser General Public License for more details.\n"
    "\n"
    "You should have received a copy of the GNU Lesser General Public\n"
    "License along with %s; if not, write to the Free Software\n"
    "Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA\n",
    program_name, program_name, program_name
#endif
    );

    return 0;
}

int show_formats(const char *opt, const char *arg)
{
    AVInputFormat *ifmt  = NULL;
    AVOutputFormat *ofmt = NULL;
    const char *last_name;

    printf("File formats:\n"
           " D. = Demuxing supported\n"
           " .E = Muxing supported\n"
           " --\n");
    last_name = "000";
    for (;;) {
        int decode = 0;
        int encode = 0;
        const char *name      = NULL;
        const char *long_name = NULL;

        while ((ofmt = av_oformat_next(ofmt))) {
            if ((name == NULL || strcmp(ofmt->name, name) < 0) &&
                strcmp(ofmt->name, last_name) > 0) {
                name      = ofmt->name;
                long_name = ofmt->long_name;
                encode    = 1;
            }
        }
        while ((ifmt = av_iformat_next(ifmt))) {
            if ((name == NULL || strcmp(ifmt->name, name) < 0) &&
                strcmp(ifmt->name, last_name) > 0) {
                name      = ifmt->name;
                long_name = ifmt->long_name;
                encode    = 0;
            }
            if (name && strcmp(ifmt->name, name) == 0)
                decode = 1;
        }
        if (name == NULL)
            break;
        last_name = name;

        printf(" %s%s %-15s %s\n",
               decode ? "D" : " ",
               encode ? "E" : " ",
               name,
            long_name ? long_name:" ");
    }
    return 0;
}

#define PRINT_CODEC_SUPPORTED(codec, field, type, list_name, term, get_name) \
    if (codec->field) {                                                      \
        const type *p = c->field;                                            \
                                                                             \
        printf("    Supported " list_name ":");                              \
        while (*p != term) {                                                 \
            get_name(*p);                                                    \
            printf(" %s", name);                                             \
            p++;                                                             \
        }                                                                    \
        printf("\n");                                                        \
    }                                                                        \

static void print_codec(const AVCodec *c)
{
    int encoder = av_codec_is_encoder(c);

    printf("%s %s [%s]:\n", encoder ? "Encoder" : "Decoder", c->name,
           c->long_name ? c->long_name : "");

    if (c->type == AVMEDIA_TYPE_VIDEO) {
        printf("    Threading capabilities: ");
        switch (c->capabilities & (CODEC_CAP_FRAME_THREADS |
                                   CODEC_CAP_SLICE_THREADS)) {
        case CODEC_CAP_FRAME_THREADS |
             CODEC_CAP_SLICE_THREADS: printf("frame and slice"); break;
        case CODEC_CAP_FRAME_THREADS: printf("frame");           break;
        case CODEC_CAP_SLICE_THREADS: printf("slice");           break;
        default:                      printf("no");              break;
        }
        printf("\n");
    }

    if (c->supported_framerates) {
        const AVRational *fps = c->supported_framerates;

        printf("    Supported framerates:");
        while (fps->num) {
            printf(" %d/%d", fps->num, fps->den);
            fps++;
        }
        printf("\n");
    }
    PRINT_CODEC_SUPPORTED(c, pix_fmts, enum PixelFormat, "pixel formats",
                          PIX_FMT_NONE, GET_PIX_FMT_NAME);
    PRINT_CODEC_SUPPORTED(c, supported_samplerates, int, "sample rates", 0,
                          GET_SAMPLE_RATE_NAME);
    PRINT_CODEC_SUPPORTED(c, sample_fmts, enum AVSampleFormat, "sample formats",
                          AV_SAMPLE_FMT_NONE, GET_SAMPLE_FMT_NAME);
    PRINT_CODEC_SUPPORTED(c, channel_layouts, uint64_t, "channel layouts",
                          0, GET_CH_LAYOUT_DESC);

    if (c->priv_class) {
        show_help_children(c->priv_class,
                           AV_OPT_FLAG_ENCODING_PARAM |
                           AV_OPT_FLAG_DECODING_PARAM);
    }
}

static char get_media_type_char(enum AVMediaType type)
{
    switch (type) {
        case AVMEDIA_TYPE_VIDEO:    return 'V';
        case AVMEDIA_TYPE_AUDIO:    return 'A';
        case AVMEDIA_TYPE_SUBTITLE: return 'S';
        default:                    return '?';
    }
}

static const AVCodec *next_codec_for_id(enum AVCodecID id, const AVCodec *prev,
                                        int encoder)
{
    while ((prev = av_codec_next(prev))) {
        if (prev->id == id &&
            (encoder ? av_codec_is_encoder(prev) : av_codec_is_decoder(prev)))
            return prev;
    }
    return NULL;
}

static void print_codecs_for_id(enum AVCodecID id, int encoder)
{
    const AVCodec *codec = NULL;

    printf(" (%s: ", encoder ? "encoders" : "decoders");

    while ((codec = next_codec_for_id(id, codec, encoder)))
        printf("%s ", codec->name);

    printf(")");
}

int show_codecs(const char *opt, const char *arg)
{
    const AVCodecDescriptor *desc = NULL;

    printf("Codecs:\n"
           " D... = Decoding supported\n"
           " .E.. = Encoding supported\n"
           " ..V. = Video codec\n"
           " ..A. = Audio codec\n"
           " ..S. = Subtitle codec\n"
           " ...I = Intra frame-only codec\n"
           " -----\n");
    while ((desc = avcodec_descriptor_next(desc))) {
        const AVCodec *codec = NULL;

        printf(avcodec_find_decoder(desc->id) ? "D" : ".");
        printf(avcodec_find_encoder(desc->id) ? "E" : ".");

        printf("%c", get_media_type_char(desc->type));
        printf((desc->props & AV_CODEC_PROP_INTRA_ONLY) ? "I" : ".");

        printf(" %-20s %s", desc->name, desc->long_name ? desc->long_name : "");

        /* print decoders/encoders when there's more than one or their
         * names are different from codec name */
        while ((codec = next_codec_for_id(desc->id, codec, 0))) {
            if (strcmp(codec->name, desc->name)) {
                print_codecs_for_id(desc->id, 0);
                break;
            }
        }
        codec = NULL;
        while ((codec = next_codec_for_id(desc->id, codec, 1))) {
            if (strcmp(codec->name, desc->name)) {
                print_codecs_for_id(desc->id, 1);
                break;
            }
        }

        printf("\n");
    }
    return 0;
}

static void print_codecs(int encoder)
{
    const AVCodecDescriptor *desc = NULL;

    printf("%s:\n"
           " V... = Video\n"
           " A... = Audio\n"
           " S... = Subtitle\n"
           " .F.. = Frame-level multithreading\n"
           " ..S. = Slice-level multithreading\n"
           " ...X = Codec is experimental\n"
           " ---\n",
           encoder ? "Encoders" : "Decoders");
    while ((desc = avcodec_descriptor_next(desc))) {
        const AVCodec *codec = NULL;

        while ((codec = next_codec_for_id(desc->id, codec, encoder))) {
            printf("%c", get_media_type_char(desc->type));
            printf((codec->capabilities & CODEC_CAP_FRAME_THREADS) ? "F" : ".");
            printf((codec->capabilities & CODEC_CAP_SLICE_THREADS) ? "S" : ".");
            printf((codec->capabilities & CODEC_CAP_EXPERIMENTAL)  ? "X" : ".");

            printf(" %-20s %s", codec->name, codec->long_name ? codec->long_name : "");
            if (strcmp(codec->name, desc->name))
                printf(" (codec %s)", desc->name);

            printf("\n");
        }
    }
}

int show_decoders(const char *opt, const char *arg)
{
    print_codecs(0);
    return 0;
}

int show_encoders(const char *opt, const char *arg)
{
    print_codecs(1);
    return 0;
}

int show_bsfs(const char *opt, const char *arg)
{
    AVBitStreamFilter *bsf = NULL;

    printf("Bitstream filters:\n");
    while ((bsf = av_bitstream_filter_next(bsf)))
        printf("%s\n", bsf->name);
    printf("\n");
    return 0;
}

int show_protocols(const char *opt, const char *arg)
{
    void *opaque = NULL;
    const char *name;

    printf("Supported file protocols:\n"
           "Input:\n");
    while ((name = avio_enum_protocols(&opaque, 0)))
        printf("%s\n", name);
    printf("Output:\n");
    while ((name = avio_enum_protocols(&opaque, 1)))
        printf("%s\n", name);
    return 0;
}

int show_filters(const char *opt, const char *arg)
{
    AVFilter av_unused(**filter) = NULL;

    printf("Filters:\n");
#if CONFIG_AVFILTER
    while ((filter = av_filter_next(filter)) && *filter)
        printf("%-16s %s\n", (*filter)->name, (*filter)->description);
#endif
    return 0;
}

int show_pix_fmts(const char *opt, const char *arg)
{
    enum PixelFormat pix_fmt;

    printf("Pixel formats:\n"
           "I.... = Supported Input  format for conversion\n"
           ".O... = Supported Output format for conversion\n"
           "..H.. = Hardware accelerated format\n"
           "...P. = Paletted format\n"
           "....B = Bitstream format\n"
           "FLAGS NAME            NB_COMPONENTS BITS_PER_PIXEL\n"
           "-----\n");

#if !CONFIG_SWSCALE
#   define sws_isSupportedInput(x)  0
#   define sws_isSupportedOutput(x) 0
#endif

    for (pix_fmt = 0; pix_fmt < PIX_FMT_NB; pix_fmt++) {
        const AVPixFmtDescriptor *pix_desc = &av_pix_fmt_descriptors[pix_fmt];
        printf("%c%c%c%c%c %-16s       %d            %2d\n",
               sws_isSupportedInput (pix_fmt)      ? 'I' : '.',
               sws_isSupportedOutput(pix_fmt)      ? 'O' : '.',
               pix_desc->flags & PIX_FMT_HWACCEL   ? 'H' : '.',
               pix_desc->flags & PIX_FMT_PAL       ? 'P' : '.',
               pix_desc->flags & PIX_FMT_BITSTREAM ? 'B' : '.',
               pix_desc->name,
               pix_desc->nb_components,
               av_get_bits_per_pixel(pix_desc));
    }
    return 0;
}

int show_sample_fmts(const char *opt, const char *arg)
{
    int i;
    char fmt_str[128];
    for (i = -1; i < AV_SAMPLE_FMT_NB; i++)
        printf("%s\n", av_get_sample_fmt_string(fmt_str, sizeof(fmt_str), i));
    return 0;
}

static void show_help_codec(const char *name, int encoder)
{
    const AVCodecDescriptor *desc;
    const AVCodec *codec;

    if (!name) {
        av_log(NULL, AV_LOG_ERROR, "No codec name specified.\n");
        return;
    }

    codec = encoder ? avcodec_find_encoder_by_name(name) :
                      avcodec_find_decoder_by_name(name);

    if (codec)
        print_codec(codec);
    else if ((desc = avcodec_descriptor_get_by_name(name))) {
        int printed = 0;

        while ((codec = next_codec_for_id(desc->id, codec, encoder))) {
            printed = 1;
            print_codec(codec);
        }

        if (!printed) {
            av_log(NULL, AV_LOG_ERROR, "Codec '%s' is known to Libav, "
                   "but no %s for it are available. Libav might need to be "
                   "recompiled with additional external libraries.\n",
                   name, encoder ? "encoders" : "decoders");
        }
    } else {
        av_log(NULL, AV_LOG_ERROR, "Codec '%s' is not recognized by Libav.\n",
               name);
    }
}

static void show_help_demuxer(const char *name)
{
    const AVInputFormat *fmt = av_find_input_format(name);

    if (!fmt) {
        av_log(NULL, AV_LOG_ERROR, "Unknown format '%s'.\n", name);
        return;
    }

    printf("Demuxer %s [%s]:\n", fmt->name, fmt->long_name);

    if (fmt->extensions)
        printf("    Common extensions: %s.\n", fmt->extensions);

    if (fmt->priv_class)
        show_help_children(fmt->priv_class, AV_OPT_FLAG_DECODING_PARAM);
}

static void show_help_muxer(const char *name)
{
    const AVCodecDescriptor *desc;
    const AVOutputFormat *fmt = av_guess_format(name, NULL, NULL);

    if (!fmt) {
        av_log(NULL, AV_LOG_ERROR, "Unknown format '%s'.\n", name);
        return;
    }

    printf("Muxer %s [%s]:\n", fmt->name, fmt->long_name);

    if (fmt->extensions)
        printf("    Common extensions: %s.\n", fmt->extensions);
    if (fmt->mime_type)
        printf("    Mime type: %s.\n", fmt->mime_type);
    if (fmt->video_codec != AV_CODEC_ID_NONE &&
        (desc = avcodec_descriptor_get(fmt->video_codec))) {
        printf("    Default video codec: %s.\n", desc->name);
    }
    if (fmt->audio_codec != AV_CODEC_ID_NONE &&
        (desc = avcodec_descriptor_get(fmt->audio_codec))) {
        printf("    Default audio codec: %s.\n", desc->name);
    }
    if (fmt->subtitle_codec != AV_CODEC_ID_NONE &&
        (desc = avcodec_descriptor_get(fmt->subtitle_codec))) {
        printf("    Default subtitle codec: %s.\n", desc->name);
    }

    if (fmt->priv_class)
        show_help_children(fmt->priv_class, AV_OPT_FLAG_ENCODING_PARAM);
}

int show_help(const char *opt, const char *arg)
{
    char *topic, *par;
    av_log_set_callback(log_callback_help);

    topic = av_strdup(arg ? arg : "");
    par = strchr(topic, '=');
    if (par)
        *par++ = 0;

    if (!*topic) {
        show_help_default(topic, par);
    } else if (!strcmp(topic, "decoder")) {
        show_help_codec(par, 0);
    } else if (!strcmp(topic, "encoder")) {
        show_help_codec(par, 1);
    } else if (!strcmp(topic, "demuxer")) {
        show_help_demuxer(par);
    } else if (!strcmp(topic, "muxer")) {
        show_help_muxer(par);
    } else {
        show_help_default(topic, par);
    }

    av_freep(&topic);
    return 0;
}

int read_yesno(void)
{
    int c = getchar();
    int yesno = (toupper(c) == 'Y');

    while (c != '\n' && c != EOF)
        c = getchar();

    return yesno;
}

int cmdutils_read_file(const char *filename, char **bufptr, size_t *size)
{
    int ret;
    FILE *f = fopen(filename, "rb");

    if (!f) {
        av_log(NULL, AV_LOG_ERROR, "Cannot read file '%s': %s\n", filename,
               strerror(errno));
        return AVERROR(errno);
    }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *bufptr = av_malloc(*size + 1);
    if (!*bufptr) {
        av_log(NULL, AV_LOG_ERROR, "Could not allocate file buffer\n");
        fclose(f);
        return AVERROR(ENOMEM);
    }
    ret = fread(*bufptr, 1, *size, f);
    if (ret < *size) {
        av_free(*bufptr);
        if (ferror(f)) {
            av_log(NULL, AV_LOG_ERROR, "Error while reading file '%s': %s\n",
                   filename, strerror(errno));
            ret = AVERROR(errno);
        } else
            ret = AVERROR_EOF;
    } else {
        ret = 0;
        (*bufptr)[*size++] = '\0';
    }

    fclose(f);
    return ret;
}

void init_pts_correction(PtsCorrectionContext *ctx)
{
    ctx->num_faulty_pts = ctx->num_faulty_dts = 0;
    ctx->last_pts = ctx->last_dts = INT64_MIN;
}

int64_t guess_correct_pts(PtsCorrectionContext *ctx, int64_t reordered_pts,
                          int64_t dts)
{
    int64_t pts = AV_NOPTS_VALUE;

    if (dts != AV_NOPTS_VALUE) {
        ctx->num_faulty_dts += dts <= ctx->last_dts;
        ctx->last_dts = dts;
    }
    if (reordered_pts != AV_NOPTS_VALUE) {
        ctx->num_faulty_pts += reordered_pts <= ctx->last_pts;
        ctx->last_pts = reordered_pts;
    }
    if ((ctx->num_faulty_pts<=ctx->num_faulty_dts || dts == AV_NOPTS_VALUE)
        && reordered_pts != AV_NOPTS_VALUE)
        pts = reordered_pts;
    else
        pts = dts;

    return pts;
}

FILE *get_preset_file(char *filename, size_t filename_size,
                      const char *preset_name, int is_path,
                      const char *codec_name)
{
    FILE *f = NULL;
    int i;
    const char *base[3] = { getenv("AVCONV_DATADIR"),
                            getenv("HOME"),
                            AVCONV_DATADIR, };

    if (is_path) {
        av_strlcpy(filename, preset_name, filename_size);
        f = fopen(filename, "r");
    } else {
        for (i = 0; i < 3 && !f; i++) {
            if (!base[i])
                continue;
            snprintf(filename, filename_size, "%s%s/%s.avpreset", base[i],
                     i != 1 ? "" : "/.avconv", preset_name);
            f = fopen(filename, "r");
            if (!f && codec_name) {
                snprintf(filename, filename_size,
                         "%s%s/%s-%s.avpreset",
                         base[i], i != 1 ? "" : "/.avconv", codec_name,
                         preset_name);
                f = fopen(filename, "r");
            }
        }
    }

    return f;
}

int check_stream_specifier(AVFormatContext *s, AVStream *st, const char *spec)
{
    if (*spec <= '9' && *spec >= '0') /* opt:index */
        return strtol(spec, NULL, 0) == st->index;
    else if (*spec == 'v' || *spec == 'a' || *spec == 's' || *spec == 'd' ||
             *spec == 't') { /* opt:[vasdt] */
        enum AVMediaType type;

        switch (*spec++) {
        case 'v': type = AVMEDIA_TYPE_VIDEO;      break;
        case 'a': type = AVMEDIA_TYPE_AUDIO;      break;
        case 's': type = AVMEDIA_TYPE_SUBTITLE;   break;
        case 'd': type = AVMEDIA_TYPE_DATA;       break;
        case 't': type = AVMEDIA_TYPE_ATTACHMENT; break;
        default:  av_assert0(0);
        }
        if (type != st->codec->codec_type)
            return 0;
        if (*spec++ == ':') { /* possibly followed by :index */
            int i, index = strtol(spec, NULL, 0);
            for (i = 0; i < s->nb_streams; i++)
                if (s->streams[i]->codec->codec_type == type && index-- == 0)
                   return i == st->index;
            return 0;
        }
        return 1;
    } else if (*spec == 'p' && *(spec + 1) == ':') {
        int prog_id, i, j;
        char *endptr;
        spec += 2;
        prog_id = strtol(spec, &endptr, 0);
        for (i = 0; i < s->nb_programs; i++) {
            if (s->programs[i]->id != prog_id)
                continue;

            if (*endptr++ == ':') {
                int stream_idx = strtol(endptr, NULL, 0);
                return stream_idx >= 0 &&
                    stream_idx < s->programs[i]->nb_stream_indexes &&
                    st->index == s->programs[i]->stream_index[stream_idx];
            }

            for (j = 0; j < s->programs[i]->nb_stream_indexes; j++)
                if (st->index == s->programs[i]->stream_index[j])
                    return 1;
        }
        return 0;
    } else if (!*spec) /* empty specifier, matches everything */
        return 1;

    av_log(s, AV_LOG_ERROR, "Invalid stream specifier: %s.\n", spec);
    return AVERROR(EINVAL);
}

AVDictionary *filter_codec_opts(AVDictionary *opts, enum AVCodecID codec_id,
                                AVFormatContext *s, AVStream *st, AVCodec *codec)
{
    AVDictionary    *ret = NULL;
    AVDictionaryEntry *t = NULL;
    int            flags = s->oformat ? AV_OPT_FLAG_ENCODING_PARAM
                                      : AV_OPT_FLAG_DECODING_PARAM;
    char          prefix = 0;
    const AVClass    *cc = avcodec_get_class();

    if (!codec)
        codec            = s->oformat ? avcodec_find_encoder(codec_id)
                                      : avcodec_find_decoder(codec_id);
    if (!codec)
        return NULL;

    switch (codec->type) {
    case AVMEDIA_TYPE_VIDEO:
        prefix  = 'v';
        flags  |= AV_OPT_FLAG_VIDEO_PARAM;
        break;
    case AVMEDIA_TYPE_AUDIO:
        prefix  = 'a';
        flags  |= AV_OPT_FLAG_AUDIO_PARAM;
        break;
    case AVMEDIA_TYPE_SUBTITLE:
        prefix  = 's';
        flags  |= AV_OPT_FLAG_SUBTITLE_PARAM;
        break;
    }

    while (t = av_dict_get(opts, "", t, AV_DICT_IGNORE_SUFFIX)) {
        char *p = strchr(t->key, ':');

        /* check stream specification in opt name */
        if (p)
            switch (check_stream_specifier(s, st, p + 1)) {
            case  1: *p = 0; break;
            case  0:         continue;
            default:         return NULL;
            }

        if (av_opt_find(&cc, t->key, NULL, flags, AV_OPT_SEARCH_FAKE_OBJ) ||
            (codec && codec->priv_class &&
             av_opt_find(&codec->priv_class, t->key, NULL, flags,
                         AV_OPT_SEARCH_FAKE_OBJ)))
            av_dict_set(&ret, t->key, t->value, 0);
        else if (t->key[0] == prefix &&
                 av_opt_find(&cc, t->key + 1, NULL, flags,
                             AV_OPT_SEARCH_FAKE_OBJ))
            av_dict_set(&ret, t->key + 1, t->value, 0);

        if (p)
            *p = ':';
    }
    return ret;
}

AVDictionary **setup_find_stream_info_opts(AVFormatContext *s,
                                           AVDictionary *codec_opts)
{
    int i;
    AVDictionary **opts;

    if (!s->nb_streams)
        return NULL;
    opts = av_mallocz(s->nb_streams * sizeof(*opts));
    if (!opts) {
        av_log(NULL, AV_LOG_ERROR,
               "Could not alloc memory for stream options.\n");
        return NULL;
    }
    for (i = 0; i < s->nb_streams; i++)
        opts[i] = filter_codec_opts(codec_opts, s->streams[i]->codec->codec_id,
                                    s, s->streams[i], NULL);
    return opts;
}

void *grow_array(void *array, int elem_size, int *size, int new_size)
{
    if (new_size >= INT_MAX / elem_size) {
        av_log(NULL, AV_LOG_ERROR, "Array too big.\n");
        exit_program(1);
    }
    if (*size < new_size) {
        uint8_t *tmp = av_realloc(array, new_size*elem_size);
        if (!tmp) {
            av_log(NULL, AV_LOG_ERROR, "Could not alloc buffer.\n");
            exit_program(1);
        }
        memset(tmp + *size*elem_size, 0, (new_size-*size) * elem_size);
        *size = new_size;
        return tmp;
    }
    return array;
}

static int alloc_buffer(FrameBuffer **pool, AVCodecContext *s, FrameBuffer **pbuf)
{
    FrameBuffer  *buf = av_mallocz(sizeof(*buf));
    int i, ret;
    const int pixel_size = av_pix_fmt_descriptors[s->pix_fmt].comp[0].step_minus1+1;
    int h_chroma_shift, v_chroma_shift;
    int edge = 32; // XXX should be avcodec_get_edge_width(), but that fails on svq1
    int w = s->width, h = s->height;

    if (!buf)
        return AVERROR(ENOMEM);

    if (!(s->flags & CODEC_FLAG_EMU_EDGE)) {
        w += 2*edge;
        h += 2*edge;
    }

    avcodec_align_dimensions(s, &w, &h);
    if ((ret = av_image_alloc(buf->base, buf->linesize, w, h,
                              s->pix_fmt, 32)) < 0) {
        av_freep(&buf);
        return ret;
    }
    /* XXX this shouldn't be needed, but some tests break without this line
     * those decoders are buggy and need to be fixed.
     * the following tests fail:
     * cdgraphics, ansi, aasc, fraps-v1, qtrle-1bit
     */
    memset(buf->base[0], 128, ret);

    avcodec_get_chroma_sub_sample(s->pix_fmt, &h_chroma_shift, &v_chroma_shift);
    for (i = 0; i < FF_ARRAY_ELEMS(buf->data); i++) {
        const int h_shift = i==0 ? 0 : h_chroma_shift;
        const int v_shift = i==0 ? 0 : v_chroma_shift;
        if (s->flags & CODEC_FLAG_EMU_EDGE)
            buf->data[i] = buf->base[i];
        else
            buf->data[i] = buf->base[i] +
                           FFALIGN((buf->linesize[i]*edge >> v_shift) +
                                   (pixel_size*edge >> h_shift), 32);
    }
    buf->w       = s->width;
    buf->h       = s->height;
    buf->pix_fmt = s->pix_fmt;
    buf->pool    = pool;

    *pbuf = buf;
    return 0;
}

int codec_get_buffer(AVCodecContext *s, AVFrame *frame)
{
    FrameBuffer **pool = s->opaque;
    FrameBuffer *buf;
    int ret, i;

    if (!*pool && (ret = alloc_buffer(pool, s, pool)) < 0)
        return ret;

    buf              = *pool;
    *pool            = buf->next;
    buf->next        = NULL;
    if (buf->w != s->width || buf->h != s->height || buf->pix_fmt != s->pix_fmt) {
        av_freep(&buf->base[0]);
        av_free(buf);
        if ((ret = alloc_buffer(pool, s, &buf)) < 0)
            return ret;
    }
    buf->refcount++;

    frame->opaque        = buf;
    frame->type          = FF_BUFFER_TYPE_USER;
    frame->extended_data = frame->data;
    frame->pkt_pts       = s->pkt ? s->pkt->pts : AV_NOPTS_VALUE;
    frame->width         = buf->w;
    frame->height        = buf->h;
    frame->format        = buf->pix_fmt;
    frame->sample_aspect_ratio = s->sample_aspect_ratio;

    for (i = 0; i < FF_ARRAY_ELEMS(buf->data); i++) {
        frame->base[i]     = buf->base[i];  // XXX h264.c uses base though it shouldn't
        frame->data[i]     = buf->data[i];
        frame->linesize[i] = buf->linesize[i];
    }

    return 0;
}

static void unref_buffer(FrameBuffer *buf)
{
    FrameBuffer **pool = buf->pool;

    av_assert0(buf->refcount);
    buf->refcount--;
    if (!buf->refcount) {
        buf->next = *pool;
        *pool = buf;
    }
}

void codec_release_buffer(AVCodecContext *s, AVFrame *frame)
{
    FrameBuffer *buf = frame->opaque;
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(frame->data); i++)
        frame->data[i] = NULL;

    unref_buffer(buf);
}

void filter_release_buffer(AVFilterBuffer *fb)
{
    FrameBuffer *buf = fb->priv;
    av_free(fb);
    unref_buffer(buf);
}

void free_buffer_pool(FrameBuffer **pool)
{
    FrameBuffer *buf = *pool;
    while (buf) {
        *pool = buf->next;
        av_freep(&buf->base[0]);
        av_free(buf);
        buf = *pool;
    }
}
