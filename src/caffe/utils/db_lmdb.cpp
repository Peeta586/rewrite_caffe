#ifdef USE_LMDB
#include "caffe/util/db_lmdb.hpp"
#include <sys/stat.h> 
#include <string> 

namespace caffe {
namespace db {
    void LMDB::Open(const string& source, Mode mode){
        MDB_CHECK(mdb_env_create(&mdb_env_)); // 创建数据库环境handle， 获取handle
        if(mode == NEW){
            CHECK_EQ(mkdir(source.c_str(), 0774), 0) << mkdir <<source << "failed";
        }
        int flags = 0;
        if (mode == READ){
            flags = MDB_RDONLY | MDB_NOTLS;
        }
        /**
         * 打开数据库的环境handle
         * int mdb_env_open  ( MDB_env *  env,  
                    const char *  path,  
                    unsigned int  flags,  
                    mdb_mode_t  mode //权限 
                    )
         */
        int rc = mdb_env_open(mdb_env_, source.c_str(), flags, 0664);

    }

} // namespace db
} // namespace caffe


#endif // USE_LMDB