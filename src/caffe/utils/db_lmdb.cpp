#ifdef USE_LMDB
#include "caffe/util/db_lmdb.hpp"
#include <sys/stat.h> 
#include <string> 

/** 数据库执行要素：
 *  先有环境handle,整个数据库环境（管理）mdb_env_
 *  再有数据库环境下的事务，（操作） mdb_txn handle
 *          mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn)
 *          mdb_txn_abort(mdb_txn);
 *  这个事务是开启那个数据库， （数据）mdb_dbi_ handle
 *          mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_)
 *          mdb_dbi_close(mdb_env_, mdb_dbi);
 *  打开数据库开始操作存储数据的时候，需要cursor 游标handle （读取指针）
 */


namespace caffe {
namespace db {
    void LMDB::Open(const string& source, Mode mode){
        MDB_CHECK(mdb_env_create(&mdb_env_)); // 创建数据库环境handle， 获取handle
        if(mode == NEW){
            CHECK_EQ(mkdir(source.c_str(), 0774), 0) << "mkdir " <<source << "failed";
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
    #ifndef ALLOW_LMDB_NOLOCK
        MDB_CHECK(rc);
    #else
        if(rc == EACCES){
            LOG(WARNING) << "Permission denied. Trying with MDB_NNOLOCK...";
            // Close and re-open environment handle
            mdb_env_close(mdb_env_);
            MDB_CHECK(mdb_env_create(&mdb_env_));
            // Try again with MDB_NOLOCK
            flags |= MDB_NOLOCK;
            MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
        }else{
            MDB_CHECK(rc);
        }
    #endif
        LOG_IF(INFO, Caffe::root_solver()) << "Opened lmdb" << source;
    }

    LMDBCursor* LMDB::NewCursor(){
        /** @brief Opaque structure for a transaction handle.
         * MDB_txn* mdb_txn ， 开启数据库前需要先创建一个事务，
         * All database operations require a transaction handle. Transactions may be
         * read-only or read-write.
         */
        MDB_txn*  mdb_txn;
        MDB_cursor* mdb_cursor;
        MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn)); // 
        MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));// 开启数据库// mdb_dbi_：A handle for an individual database in the DB environment
        MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
        return new LMDBCursor(mdb_txn, mdb_cursor);
    }

    LMDBTransaction* LMDB::NewTransaction(){
        return new LMDBTransaction(mdb_env_);
    }

    // 事务的两个操作 put, commit;
    void LMDBTransaction::Put(const string& key, const string& value){
        
        keys.push_back(key);
        values.push_back(value);
    }

    void LMDBTransaction::Commit(){
        MDB_dbi mdb_dbi;
        MDB_val mdb_key, mdb_data;
        MDB_txn * mdb_txn;

        // initialize MDB variables
        MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn));
        MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi));

        for(int i =0; i< keys.size(); i++){
            mdb_key.mv_size = keys[i].size();
            mdb_key.mv_data = const_cast<char*>(keys[i].data());
            mdb_data.mv_size = values[i].size();
            // string类型存储图片数据，将其转化为char*类型，则每个元素就是一个像素
            mdb_data.mv_data = const_cast<char*>(values[i].data());

            // Add data to the transaction
            int put_rc = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
            if (put_rc == MDB_MAP_FULL){ // 如果满了
                // Out of memory -double the map size and retry
                mdb_txn_abort(mdb_txn); // 先中断
                mdb_dbi_close(mdb_env_, mdb_dbi); // 关闭数据库
                DoubleMapSize(); // 扩充数据库空间两倍后，再重试
                Commit();
                return;
            }
            // May have failed for some other reason
            MDB_CHECK(put_rc);
        }

        // commit the transaction
        int commit_rc = mdb_txn_commit(mdb_txn);
        if (commit_rc == MDB_MAP_FULL){ 
            // out of memory - double the map size and retry
            mdb_dbi_close(mdb_env_, mdb_dbi);
            DoubleMapSize();
            Commit();
            return;
        }
        // may have failed for some other reason
        MDB_CHECK(commit_rc);

        // cleanup after successful commit
        mdb_dbi_close(mdb_env_, mdb_dbi);
        keys.clear();
        values.clear();
    }
    
    void LMDBTransaction::DoubleMapSize(){
        struct MDB_envinfo current_info;
        // 获取大小信息
        MDB_CHECK(mdb_env_info(mdb_env_, &current_info));
        size_t new_size = current_info.me_mapsize * 2;
        DLOG(INFO)<<"doubling lmdb map size to "<< (new_size >> 20) <<"MB ...";
        // 扩容
        MDB_CHECK(mdb_env_set_mapsize(mdb_env_, new_size));
    }

} // namespace db
} // namespace caffe


#endif // USE_LMDB