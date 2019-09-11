#include "caffe/util/db.hpp"
#include "caffe/util/db_leveldb.hpp"
#include "caffe/util/db_lmdb.hpp"

#include <string>

namespace caffe{ 
namespace db {
    DB* GetDB(DataParameter::DB backend) {
        switch (backend)
        {
        #ifdef USE_LEVELDB
        case DataParameter_DB_LEVELDB:
            return new LevelDB();
        #endif  // USE LEVELDB
        #ifdef USE_LMDB
        case DataParameter_DB_LMDB: 
            return new LMDB();
        #endif  // USE LMDB
        default:
            LOG(FATAL) << "Unknown database backend";
            return NULL;
        }
    }

} // namespace db

} // namespace caffe